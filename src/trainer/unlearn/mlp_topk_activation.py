import math
import re

import torch

from trainer.unlearn.grad_diff import GradDiff


class MLPTopKActivation(GradDiff):
    def __init__(
        self,
        topk_percent=1.0,
        min_topk=1,
        mlp_module_pattern=r"(^|\.)model\.layers\.(?:[8-9]|1[0-5])\.mlp(\.|$)",
        exclude_module_pattern=r"(self_attn|attention)",
        activation_regularization="l1",
        gamma=1.0,
        alpha=1.0,
        retain_loss_type="NLL",
        *args,
        **kwargs,
    ):
        super().__init__(
            gamma=gamma,
            alpha=alpha,
            retain_loss_type=retain_loss_type,
            *args,
            **kwargs,
        )
        if topk_percent <= 0 or topk_percent > 100:
            raise ValueError("topk_percent must be in (0, 100].")
        if min_topk < 1:
            raise ValueError("min_topk must be >= 1.")
        if activation_regularization not in {"l1", "l2"}:
            raise ValueError("activation_regularization must be one of {'l1', 'l2'}.")

        self.topk_percent = topk_percent
        self.min_topk = min_topk
        self.mlp_module_pattern = re.compile(mlp_module_pattern)
        self.exclude_module_pattern = re.compile(exclude_module_pattern)
        self.activation_regularization = activation_regularization
        self._cached_mlp_module_names = None

    def _unwrap_model(self, model):
        while hasattr(model, "module"):
            model = model.module
        return model

    def _get_mlp_modules(self, model):
        if self._cached_mlp_module_names is None:
            base_model = self._unwrap_model(model)
            matched_names = []
            for name, _ in base_model.named_modules():
                if not name:
                    continue
                if not self.mlp_module_pattern.search(name):
                    continue
                if self.exclude_module_pattern.search(name):
                    continue
                matched_names.append(name)

            if not matched_names:
                raise ValueError(
                    "No MLP modules found. Try adjusting mlp_module_pattern in method_args."
                )

            self._cached_mlp_module_names = matched_names

        base_model = self._unwrap_model(model)
        module_dict = dict(base_model.named_modules())
        return [module_dict[name] for name in self._cached_mlp_module_names]

    def _forward_with_mlp_activations(self, model, inputs):
        modules = self._get_mlp_modules(model)
        activations = []
        handles = []

        def _hook(_module, _inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            if torch.is_tensor(output):
                activations.append(output)

        for module in modules:
            handles.append(module.register_forward_hook(_hook))

        try:
            outputs = model(**inputs)
        finally:
            for handle in handles:
                handle.remove()

        return activations, outputs

    def _get_topk_activation_loss(self, activations, labels=None):
        if not activations:
            raise ValueError("No MLP activations were captured for computing loss.")

        reg_values = []
        token_mask = None
        if labels is not None:
            token_mask = labels != -100

        for activation in activations:
            if not torch.is_tensor(activation):
                continue

            per_element = (
                activation.abs()
                if self.activation_regularization == "l1"
                else activation.pow(2)
            )

            if (
                token_mask is not None
                and activation.dim() == 3
                and activation.shape[0] == token_mask.shape[0]
                and activation.shape[1] == token_mask.shape[1]
                and token_mask.any()
            ):
                reg_values.append(per_element[token_mask].reshape(-1))
            else:
                reg_values.append(per_element.reshape(-1))

        if not reg_values:
            raise ValueError("Captured MLP activations are empty.")

        flat_values = torch.cat(reg_values, dim=0)
        topk = max(
            self.min_topk,
            int(math.ceil(flat_values.numel() * (self.topk_percent / 100.0))),
        )
        topk = min(topk, flat_values.numel())

        return torch.topk(flat_values, k=topk, largest=True, sorted=False).values.mean()

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }

        forget_activations, forget_outputs = self._forward_with_mlp_activations(
            model, forget_inputs
        )
        forget_loss = self._get_topk_activation_loss(
            forget_activations, labels=forget_inputs.get("labels")
        )

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        self.log(
            {
                "forget_loss": forget_loss.detach().item(),
                "retain_loss": retain_loss.detach().item(),
            }
        )

        loss = self.gamma * forget_loss + self.alpha * retain_loss

        return (loss, forget_outputs) if return_outputs else loss
