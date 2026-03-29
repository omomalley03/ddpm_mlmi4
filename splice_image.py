

from PIL import Image

def delete_every_second_column(input_path, output_path, num_cols=9):
    img = Image.open(input_path)
    width, height = img.size

    tile_w = width // num_cols

    # Fix: crop(left, upper, right, lower) — was previously swapped
    tiles = []
    for i in range(num_cols):
        x_start = i * tile_w
        x_end = x_start + tile_w
        tile = img.crop((x_start, 0, x_end, height))  # ← was (0, x_start, height, x_end)
        tiles.append(tile)

    # Keep every other column: 0, 2, 4, 6, 8
    kept_tiles = [tiles[i] for i in [0, 2, 4, 5, 6, 8]]

    new_width = tile_w * len(kept_tiles)
    result = Image.new(img.mode, (new_width, height))
    for i, tile in enumerate(kept_tiles):
        result.paste(tile, (i * tile_w, 0))

    result.save(output_path)
    print(f"Saved {len(kept_tiles)} columns to: {output_path}")

if __name__ == "__main__":
    input_path = "/rds/user/dpc49/hpc-work/MLMI4/interpolation_compare/pair_04_comparison.png"
    output_path = "interpolation_thinned.png"
    delete_every_second_column(input_path, output_path, num_cols=9)