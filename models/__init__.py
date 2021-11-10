# python3.7
"""Collects all available models together."""

from .model_zoo import MODEL_ZOO
from .pggan_generator import PGGANGenerator
from .pggan_discriminator import PGGANDiscriminator
from .stylegan_generator import StyleGANGenerator
from .stylegan_discriminator import StyleGANDiscriminator
from .stylegan2_generator import StyleGAN2Generator
from .stylegan2_discriminator import StyleGAN2Discriminator
from .networks_stylegan3 import Generator as StyleGAN3Generator
from .networks_stylegan3 import Discriminator as StyleGAN3Discriminator

__all__ = [
    'MODEL_ZOO', 'PGGANGenerator', 'PGGANDiscriminator', 'StyleGANGenerator',
    'StyleGANDiscriminator', 'StyleGAN2Generator', 'StyleGAN2Discriminator', 'StyleGAN3Generator', 'StyleGAN3Discriminator',
    'build_generator', 'build_discriminator', 'build_model'
]

_GAN_TYPES_ALLOWED = ['pggan', 'stylegan', 'stylegan2', 'stylegan3']
_MODULES_ALLOWED = ['generator', 'discriminator']


def build_generator(gan_type, resolution, **kwargs):
    """Builds generator by GAN type.

    Args:
        gan_type: GAN type to which the generator belong.
        resolution: Synthesis resolution.
        **kwargs: Additional arguments to build the generator.

    Raises:
        ValueError: If the `gan_type` is not supported.
        NotImplementedError: If the `gan_type` is not implemented.
    """
    if gan_type not in _GAN_TYPES_ALLOWED:
        raise ValueError(f'Invalid GAN type: `{gan_type}`!\n'
                         f'Types allowed: {_GAN_TYPES_ALLOWED}.')

    if gan_type == 'pggan':
        return PGGANGenerator(resolution, **kwargs)
    if gan_type == 'stylegan':
        return StyleGANGenerator(resolution, **kwargs)
    if gan_type == 'stylegan2':
        return StyleGAN2Generator(resolution, **kwargs)
    if gan_type == 'stylegan3':
        return StyleGAN3Generator(z_dim = 512, c_dim = 0, w_dim = 512, img_resolution = 256, img_channels = 3, mapping_kwargs = {}, channel_base = 16384, num_layers=14, num_critical=2, margin_size=10, num_fp16_res=4)
    raise NotImplementedError(f'Unsupported GAN type `{gan_type}`!')


def build_discriminator(gan_type, resolution, **kwargs):
    """Builds discriminator by GAN type.

    Args:
        gan_type: GAN type to which the discriminator belong.
        resolution: Synthesis resolution.
        **kwargs: Additional arguments to build the discriminator.

    Raises:
        ValueError: If the `gan_type` is not supported.
        NotImplementedError: If the `gan_type` is not implemented.
    """
    if gan_type not in _GAN_TYPES_ALLOWED:
        raise ValueError(f'Invalid GAN type: `{gan_type}`!\n'
                         f'Types allowed: {_GAN_TYPES_ALLOWED}.')

    if gan_type == 'pggan':
        return PGGANDiscriminator(resolution, **kwargs)
    if gan_type == 'stylegan':
        return StyleGANDiscriminator(resolution, **kwargs)
    if gan_type == 'stylegan2':
        return StyleGAN2Discriminator(resolution, **kwargs)
    if gan_type == 'stylegan3':
        return StyleGAN3Discriminator(c_dim=0, img_resolution=256, img_channels=3)
    raise NotImplementedError(f'Unsupported GAN type `{gan_type}`!')


def build_model(gan_type, module, resolution, **kwargs):
    """Builds a GAN module (generator/discriminator/etc).

    Args:
        gan_type: GAN type to which the model belong.
        module: GAN module to build, such as generator or discrimiantor.
        resolution: Synthesis resolution.
        **kwargs: Additional arguments to build the discriminator.

    Raises:
        ValueError: If the `module` is not supported.
        NotImplementedError: If the `module` is not implemented.
    """
    if module not in _MODULES_ALLOWED:
        raise ValueError(f'Invalid module: `{module}`!\n'
                         f'Modules allowed: {_MODULES_ALLOWED}.')

    if module == 'generator':
        return build_generator(gan_type, resolution, **kwargs)
    if module == 'discriminator':
        return build_discriminator(gan_type, resolution, **kwargs)
    raise NotImplementedError(f'Unsupported module `{module}`!')


def parse_gan_type(module):
    """Parses GAN type of a given module.

    Args:
        module: The module to parse GAN type from.

    Returns:
        A string, indicating the GAN type.

    Raises:
        ValueError: If the GAN type is unknown.
    """
    if isinstance(module, (PGGANGenerator, PGGANDiscriminator)):
        return 'pggan'
    if isinstance(module, (StyleGANGenerator, StyleGANDiscriminator)):
        return 'stylegan'
    if isinstance(module, (StyleGAN2Generator, StyleGAN2Discriminator)):
        return 'stylegan2'
    if isinstance(module, (StyleGAN3Generator, StyleGAN3Discriminator)):
        return 'stylegan3'
    raise ValueError(f'Unable to parse GAN type from type `{type(module)}`!')
