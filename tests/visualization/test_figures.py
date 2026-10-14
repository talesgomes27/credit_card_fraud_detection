"""Tests for visual storytelling and figure generation module."""

from PIL import Image

from src.visualization.figures import EXPECTED_FIGURE_FILENAMES, generate_all_figures


def test_generate_all_figures_creates_300dpi_images(tmp_path):
    """Test that all 8 figures are generated at 300 DPI and non-empty."""
    output_dir = tmp_path / "figures"

    # Generate all figures on a small subset or full dataset
    created_files = generate_all_figures(output_dir=output_dir, sample_size=2000)

    assert len(created_files) == 8

    for filename in EXPECTED_FIGURE_FILENAMES:
        file_path = output_dir / filename
        assert file_path.exists(), f"Figure {filename} was not generated"
        assert file_path.stat().st_size > 5_000, f"Figure {filename} is too small / empty"

        # Check DPI and valid image structure
        with Image.open(file_path) as img:
            dpi = img.info.get("dpi", (300, 300))
            assert round(dpi[0]) >= 300, f"Figure {filename} DPI is less than 300: {dpi}"
            assert img.size[0] > 500 and img.size[1] > 300
