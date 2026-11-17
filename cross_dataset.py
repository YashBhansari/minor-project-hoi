"""Cross-dataset evaluation: relabel a UPT model's detections into another dataset's classes.

HICO-DET has 117 verbs and V-COCO has 24 actions, so a model trained on one cannot be
scored on the other until its predicted classes are translated. Only classes that have a
counterpart are translated; every other prediction is dropped, and target classes without
a counterpart score zero AP. Report cross-dataset numbers with that caveat.
"""
import torch

# (V-COCO action, HICO-DET verb) pairs that mean the same interaction, beyond the ones
# whose verb names are identical (hold, ride, hit, eat, jump, carry, throw, catch, cut,
# kick, read), which are paired automatically.
# Each pair was checked against both datasets' object lists: the two classes share at
# least one object type. "eat instr" (fork, knife, spoon) has no HICO-DET counterpart and
# is left out. The loosest pairs are "look obj"/"watch" (4 shared objects of 64) and
# "hit instr"/"swing" (swinging a bat or racket is not always a hit).
EXTRA_PAIRS = [
    ("sit instr", "sit_on"),
    ("sit instr", "sit_at"),
    ("lay instr", "lie_on"),
    ("look obj", "watch"),
    ("talk_on_phone instr", "talk_on"),
    ("work_on_computer instr", "type_on"),
    ("ski instr", "ride"),
    ("surf instr", "ride"),
    ("skateboard instr", "ride"),
    ("snowboard instr", "ride"),
    ("drink instr", "drink_with"),
    ("drink instr", "sip"),
    ("cut instr", "cut_with"),
    ("hit instr", "swing"),
]


def classes(dataset):
    """(class names, valid classes per detector object) of a pocket HICODet or VCOCO dataset."""
    if hasattr(dataset, "verbs"):
        return dataset.verbs, dataset.object_to_verb
    return dataset.actions, list(dataset.object_to_action.values())


def class_pairs(source_names, target_names):
    """[(source index, target index)] for every pair of classes that mean the same interaction."""
    def verb(name):
        return name.split()[0]
    same = {(s, t) for s in source_names for t in target_names if verb(s) == verb(t)}
    for a, b in EXTRA_PAIRS:
        for s, t in ((a, b), (b, a)):
            if s in source_names and t in target_names:
                same.add((s, t))
    unknown = [p for p in EXTRA_PAIRS if not ({(p[0], p[1]), (p[1], p[0])} & same)]
    if unknown:
        raise ValueError(f"EXTRA_PAIRS names not found in the datasets: {unknown}")
    return sorted((source_names.index(s), target_names.index(t)) for s, t in same)


class Relabel(torch.nn.Module):
    """Wrap a UPT model so its detections carry the target dataset's class indices.

    A detection is kept once per target class its predicted class is paired with, and only
    if the target dataset allows that class for the detected object.
    """

    def __init__(self, model, source_dataset, target_dataset):
        super().__init__()
        self.module = model  # the evaluation code reaches the UPT model through .module
        target_names, valid = classes(target_dataset)
        pairs = class_pairs(classes(source_dataset)[0], target_names)
        self.register_buffer("source", torch.tensor([s for s, _ in pairs]))
        self.register_buffer("target", torch.tensor([t for _, t in pairs]))
        mask = torch.zeros(len(valid), len(target_names), dtype=torch.bool)
        for obj, allowed in enumerate(valid):
            mask[obj, allowed] = True
        self.register_buffer("valid", mask)

    def forward(self, images):
        outputs = self.module(images)
        for out in outputs:
            det, pair = torch.nonzero(out["labels"][:, None] == self.source[None]).unbind(1)
            keep = self.valid[out["objects"][det], self.target[pair]]
            det, pair = det[keep], pair[keep]
            out.update(pairing=out["pairing"][:, det], scores=out["scores"][det],
                       objects=out["objects"][det], labels=self.target[pair])
        return outputs


if __name__ == "__main__":  # self-check with a fake two-detection model
    from types import SimpleNamespace as NS

    EXTRA_PAIRS = []  # the toy datasets below do not contain the real class names

    source = NS(verbs=["ride", "wash"], object_to_verb=[[0, 1], [0]])
    target = NS(actions=["hold obj", "ride instr"], object_to_action={1: [0], 2: [0, 1]})
    class Fake(torch.nn.Module):
        def forward(self, images):
            return [dict(labels=torch.tensor([0, 1, 0]), objects=torch.tensor([1, 1, 0]),
                         scores=torch.tensor([0.9, 0.8, 0.7]),
                         pairing=torch.tensor([[0, 0, 0], [1, 1, 2]]))]

    model = Relabel(Fake(), source, target)
    out = model(None)[0]
    # "wash" has no counterpart; "ride" on object 0 is not allowed by the target dataset
    assert out["labels"].tolist() == [1] and out["scores"].allclose(torch.tensor([0.9])), out
    assert out["pairing"].tolist() == [[0], [1]] and out["objects"].tolist() == [1]
    print("ok")
