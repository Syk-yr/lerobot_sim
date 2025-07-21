import torch

class Image_Features():
    def __init__(self,):
        super().__init__()

        # extract parameters from the configuration
        self.model_zoo_cfg: dict =None  # type: ignore
        self.model_name: str = "resnet18"  # type: ignore
        self.model_device: str = "cuda"  # type: ignore

        # List of Theia models - These are configured through `_prepare_theia_transformer_model` function
        default_theia_models = [
            "theia-tiny-patch16-224-cddsv",
            "theia-tiny-patch16-224-cdiv",
            "theia-small-patch16-224-cdiv",
            "theia-base-patch16-224-cdiv",
            "theia-small-patch16-224-cddsv",
            "theia-base-patch16-224-cddsv",
        ]
        # List of ResNet models - These are configured through `_prepare_resnet_model` function
        default_resnet_models = ["resnet18", "resnet34", "resnet50", "resnet101"]

       
        if self.model_zoo_cfg is None:
            if self.model_name in default_resnet_models:
                model_config = self._prepare_resnet_model(self.model_name, self.model_device)
            else:
                raise ValueError(
                    f"Model name '{self.model_name}' not found in the default model zoo configuration."
                    f" Available models: {default_theia_models + default_resnet_models}."
                )
        else:
            model_config = self.model_zoo_cfg[self.model_name]

        # Retrieve the model, preprocess and inference functions
        self._model = model_config["model"]()
        self._inference_fn = model_config["inference"]

    def act(
        self,
        image_data: torch.Tensor,
        inference_kwargs: dict | None = None,
    ) -> torch.Tensor:
        image_device = image_data.device
        # forward the images through the model
        features = self._inference_fn(self._model, image_data, **(inference_kwargs or {}))
        # move the features back to the image device
        return features.detach().to(image_device)

    """
    Helper functions.
    """

    def _prepare_resnet_model(self, model_name: str, model_device: str) -> dict:
        """Prepare the ResNet model for inference.

        Args:
            model_name: The name of the ResNet model to prepare.
            model_device: The device to store and infer the model on.

        Returns:
            A dictionary containing the model and inference functions.
        """
        from torchvision import models

        def _load_model() -> torch.nn.Module:
            """Load the ResNet model."""
            # map the model name to the weights
            resnet_weights = {
                "resnet18": "ResNet18_Weights.IMAGENET1K_V1",
                "resnet34": "ResNet34_Weights.IMAGENET1K_V1",
                "resnet50": "ResNet50_Weights.IMAGENET1K_V1",
                "resnet101": "ResNet101_Weights.IMAGENET1K_V1",
            }

            # load the model
            model = getattr(models, model_name)(weights=resnet_weights[model_name]).eval()
            return model.to(model_device)

        def _inference(model, images: torch.Tensor) -> torch.Tensor:
            """Inference the ResNet model.

            Args:
                model: The ResNet model.
                images: The preprocessed image tensor. Shape is (num_envs, channel, height, width).

            Returns:
                The extracted features tensor. Shape is (num_envs, feature_dim).
            """
            # move the image to the model device
            image_proc = images.to(model_device)
            # permute the image to (num_envs, channel, height, width)
            image_proc = image_proc.permute(0, 3, 1, 2).float() / 255.0
            # normalize the image
            mean = torch.tensor([0.485, 0.456, 0.406], device=model_device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=model_device).view(1, 3, 1, 1)
            image_proc = (image_proc - mean) / std

            # forward the image through the model
            return model(image_proc)

        # return the model, preprocess and inference functions
        return {"model": _load_model, "inference": _inference}