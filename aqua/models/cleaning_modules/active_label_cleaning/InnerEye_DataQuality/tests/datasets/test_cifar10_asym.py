#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from aqua.models.cleaning_modules.active_label_cleaning.InnerEye_DataQuality.default_paths import CIFAR10_ROOT_DIR
from aqua.models.cleaning_modules.active_label_cleaning.InnerEye_DataQuality.InnerEyeDataQuality.datasets.cifar10_asym_noise import CIFAR10AsymNoise

def test_cifar10_asym() -> None:
    CIFAR10AsymNoise(root=str(CIFAR10_ROOT_DIR), train=True, transform=None, download=True)
