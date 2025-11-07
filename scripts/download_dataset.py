#!/usr/bin/env python3
"""
Download and prepare the redsm5 dataset from HuggingFace.
"""

import os
from pathlib import Path
from datasets import load_dataset
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_redsm5_dataset(output_dir: str = "data/redsm5"):
    """
    Download the redsm5 dataset from HuggingFace and save locally.

    Args:
        output_dir: Directory to save the dataset
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading redsm5 dataset from HuggingFace...")

    try:
        # Load dataset from HuggingFace
        dataset = load_dataset("irlab-udc/redsm5")

        logger.info(f"Dataset loaded successfully!")
        logger.info(f"Dataset structure: {dataset}")

        # Save dataset to disk
        logger.info(f"Saving dataset to {output_path}...")

        # Save each split
        for split_name, split_data in dataset.items():
            split_path = output_path / f"{split_name}.json"
            split_data.to_json(split_path, orient="records", lines=True)
            logger.info(f"  - {split_name}: {len(split_data)} examples saved to {split_path}")

        # Also save in arrow format for faster loading
        dataset.save_to_disk(str(output_path / "arrow"))
        logger.info(f"Dataset saved in Arrow format to {output_path / 'arrow'}")

        # Print dataset statistics
        logger.info("\nDataset Statistics:")
        for split_name, split_data in dataset.items():
            logger.info(f"\n{split_name.upper()} Split:")
            logger.info(f"  Number of examples: {len(split_data)}")
            logger.info(f"  Features: {split_data.features}")

            # Print first example
            if len(split_data) > 0:
                logger.info(f"  First example keys: {list(split_data[0].keys())}")

        logger.info(f"\nDataset download complete! Saved to {output_path}")

    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        logger.error("Please check your internet connection and HuggingFace credentials if needed.")
        raise


if __name__ == "__main__":
    download_redsm5_dataset()
