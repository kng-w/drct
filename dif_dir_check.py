import os

# ディレクトリのパス
dir_A = "narrative_graph/data/output/Manga109_Image_Processed/panel_cropped/name-frameid_second"
dir_B = "output/name-frameid_second-sr_xl"

def collect_files(root_dir):
    file_set = set()
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".jpg"):
                # title/page_no/*.jpg に対応：ディレクトリ構造を保った相対パスを記録
                rel_path = os.path.relpath(os.path.join(root, f), root_dir)
                file_set.add(rel_path)
    return file_set

def main():
    files_A = collect_files(dir_A)
    files_B = collect_files(dir_B)

    missing = sorted(files_A - files_B)

    print(f"Total missing files: {len(missing)}")
    for f in missing:
        print(f)

    # 必要ならリストを保存
    with open("missing_files.txt", "w") as out:
        out.write("\n".join(missing))
    print("\nSaved missing file list to missing_files.txt")

if __name__ == "__main__":
    main()