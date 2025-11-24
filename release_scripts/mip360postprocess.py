import pandas as pd
import os
import sys


def create_psnr_table_from_csv(csv_path, output_md=None):
    """
    Read experiment_results.csv and create markdown tables with test PSNR, train PSNR, and GPU memory.
    Rows: scenes, Columns: clm_offload, naive_offload, no_offload
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file {csv_path} does not exist")
        return

    # Read the CSV
    df = pd.read_csv(csv_path)

    # Parse experiment names to extract scene and offload_type
    # Format: TIMESTAMP_TIMESTAMP_SCENENAME_OFFLOADTYPE
    # e.g., 20251111_231509_bicycle_no_offload
    def parse_experiment_name(exp_name):
        parts = exp_name.split("_")
        if len(parts) >= 3:
            # Last part(s) are offload type, everything before last 2 timestamps is scene
            # Find where the offload type starts (it should be at the end)
            # Possible values: "clm_offload", "naive_offload", "no_offload"

            # Try to match the offload pattern at the end
            exp_lower = exp_name.lower()
            if exp_lower.endswith("_clm_offload"):
                offload_type = "clm_offload"
                scene_part = exp_name[: -len("_clm_offload")]
            elif exp_lower.endswith("_naive_offload"):
                offload_type = "naive_offload"
                scene_part = exp_name[: -len("_naive_offload")]
            elif exp_lower.endswith("_no_offload"):
                offload_type = "no_offload"
                scene_part = exp_name[: -len("_no_offload")]
            else:
                # Fallback: assume last part is offload type
                offload_type = parts[-1]
                scene_part = "_".join(parts[:-1])

            # Remove timestamp prefix (first 2 parts: date_time)
            scene_parts = scene_part.split("_")
            if len(scene_parts) >= 3:
                scene = "_".join(scene_parts[2:])
            else:
                scene = scene_part

            return scene, offload_type
        return None, None

    # Add scene and offload_type columns
    df[["scene", "offload_type"]] = df["experiment"].apply(
        lambda x: pd.Series(parse_experiment_name(x))
    )

    # Ensure columns are in the desired order
    desired_columns = ["clm_offload", "naive_offload", "no_offload"]

    # Rename columns for better display
    column_rename = {
        "scene": "Scene",
        "clm_offload": "CLM Offload",
        "naive_offload": "Naive Offload",
        "no_offload": "No Offload",
    }

    # Create pivot table for test PSNR
    test_pivot_df = df.pivot(index="scene", columns="offload_type", values="test_psnr")
    available_columns = [col for col in desired_columns if col in test_pivot_df.columns]
    test_pivot_df = test_pivot_df[available_columns]
    test_pivot_df = test_pivot_df.round(2)
    test_pivot_df = test_pivot_df.reset_index()
    test_pivot_df = test_pivot_df.rename(columns=column_rename)

    # Create pivot table for train PSNR
    train_pivot_df = df.pivot(
        index="scene", columns="offload_type", values="train_psnr"
    )
    available_columns_train = [
        col for col in desired_columns if col in train_pivot_df.columns
    ]
    train_pivot_df = train_pivot_df[available_columns_train]
    train_pivot_df = train_pivot_df.round(2)
    train_pivot_df = train_pivot_df.reset_index()
    train_pivot_df = train_pivot_df.rename(columns=column_rename)

    # Create pivot table for max GPU memory
    memory_pivot_df = df.pivot(
        index="scene", columns="offload_type", values="max_gpu_memory_gb"
    )
    available_columns_memory = [
        col for col in desired_columns if col in memory_pivot_df.columns
    ]
    memory_pivot_df = memory_pivot_df[available_columns_memory]
    memory_pivot_df = memory_pivot_df.round(2)
    memory_pivot_df = memory_pivot_df.reset_index()
    memory_pivot_df = memory_pivot_df.rename(columns=column_rename)

    # Create pivot table for Gaussians count
    gaussians_pivot_df = df.pivot(
        index="scene", columns="offload_type", values="num_3dgs"
    )
    available_columns_gaussians = [
        col for col in desired_columns if col in gaussians_pivot_df.columns
    ]
    gaussians_pivot_df = gaussians_pivot_df[available_columns_gaussians]
    gaussians_pivot_df = gaussians_pivot_df.reset_index()
    gaussians_pivot_df = gaussians_pivot_df.rename(columns=column_rename)

    # Generate markdown tables
    test_markdown_table = test_pivot_df.to_markdown(index=False)
    train_markdown_table = train_pivot_df.to_markdown(index=False)
    memory_markdown_table = memory_pivot_df.to_markdown(index=False)
    gaussians_markdown_table = gaussians_pivot_df.to_markdown(index=False)

    # Save to file if output path specified
    if output_md is None:
        output_md = csv_path.replace(".csv", "_psnr_table.md")

    with open(output_md, "w") as f:
        f.write("# Performance Metrics by Scene and Offload Type\n\n")
        f.write("## Test PSNR\n\n")
        f.write(test_markdown_table)
        f.write("\n\n")
        f.write("## Train PSNR\n\n")
        f.write(train_markdown_table)
        f.write("\n\n")
        f.write("## Max GPU Memory (GB)\n\n")
        f.write(memory_markdown_table)
        f.write("\n\n")
        f.write("## Gaussians Count\n\n")
        f.write(gaussians_markdown_table)
        f.write("\n")

    print(f"\nSuccessfully created performance tables and saved to {output_md}")
    print("\n" + "=" * 70)
    print("Test PSNR by Scene and Offload Type")
    print("=" * 70)
    print(test_markdown_table)
    print("\n" + "=" * 70)
    print("Train PSNR by Scene and Offload Type")
    print("=" * 70)
    print(train_markdown_table)
    print("\n" + "=" * 70)
    print("Max GPU Memory (GB) by Scene and Offload Type")
    print("=" * 70)
    print(memory_markdown_table)
    print("\n" + "=" * 70)
    print("Gaussians Count by Scene and Offload Type")
    print("=" * 70)
    print(gaussians_markdown_table)
    print("=" * 70)

    return test_pivot_df, train_pivot_df, memory_pivot_df, gaussians_pivot_df


if __name__ == "__main__":
    # read args from command line
    if len(sys.argv) < 2:
        print("Usage: python mip360postprocess.py <csv_path> [output_md]")
        print("  csv_path: Path to experiment_results.csv file")
        print("  output_md: (optional) Path to output markdown file")
        print("\nExample:")
        print("  python mip360postprocess.py experiment_results.csv")
        print(
            "  python mip360postprocess.py experiment_results.csv performance_tables.md"
        )
        sys.exit(1)

    csv_path = sys.argv[1]
    output_md = sys.argv[2] if len(sys.argv) > 2 else None

    create_psnr_table_from_csv(csv_path, output_md)
