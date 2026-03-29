from trainer.utils import compute_undial_probRedistribution_loss
from trainer.unlearn.grad_diff import GradDiff


class UNDIALProbRedistribution(GradDiff):
    def __init__(self, lambda_uniform=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_uniform = lambda_uniform
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forget_inputs = inputs["forget"]
        forget_loss, forget_outputs = compute_undial_probRedistribution_loss(
            model, self.ref_model, forget_inputs, self.lambda_uniform
        )

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss
