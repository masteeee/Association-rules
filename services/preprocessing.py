import pandas as pd

import logging
import datetime

logging.basicConfig(
    filename='preprocess.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Kategori yang diklasifikasikan sebagai Makanan
FOOD_KEYWORDS = ["FOOD"]

# Kategori yang diklasifikasikan sebagai Minuman
DRINK_KEYWORDS = ["DRINK"]


def classify_category(cat):
    """Klasifikasikan kategori produk ke Makanan / Minuman / Lainnya."""
    if not isinstance(cat, str):
        return "Lainnya"
    cat_upper = cat.upper()
    if any(k in cat_upper for k in FOOD_KEYWORDS):
        return "Makanan"
    elif any(k in cat_upper for k in DRINK_KEYWORDS):
        return "Minuman"
    else:
        return "Lainnya"


def preprocess(filepath, bill_col, name_col, category_col=None):
    """
    Membaca file dan melakukan preprocessing lengkap dengan pemetaan kolom kustom.

    Returns:
        dict berisi:
          - df_raw       : DataFrame mentah (semua kolom)
          - df_clean     : DataFrame bersih (Bill Number, Product Name, Product Category)
          - basket       : Series per transaksi (daftar produk)
          - basket_matrix: DataFrame one-hot encoding (untuk FP-Growth)
          - stats        : dict statistik dasar
          - category_dist: Series distribusi Product Category
          - type_dist    : Series distribusi Makanan / Minuman / Lainnya
          - top_products : Series produk terlaris (top 15)
    """
    # ── Baca file ──────────────────────────────────────────────────────────────
    logger.info("Starting preprocessing for file: %s", filepath)
    if filepath.lower().endswith(".csv"):
        try:
            # Coba deteksi separator secara otomatis (misal koma atau titik koma)
            df_raw = pd.read_csv(filepath, sep=None, engine="python", encoding="utf-8")
        except Exception:
            # Fallback jika gagal deteksi otomatis
            df_raw = pd.read_csv(filepath, encoding="utf-8")
    else:
        df_raw = pd.read_excel(filepath)

    # Bersihkan nama kolom
    df_raw.columns = df_raw.columns.str.strip()

    # Simpan salinan asli untuk referensi (raw data sebelum filter menu paket)
    df_raw_original = df_raw.copy()

    # Hitung total baris raw sebelum filter (untuk statistik)
    initial_raw_count = len(df_raw_original)

    # Lanjutkan dengan proses selanjutnya menggunakan df_raw (yang akan difilter)
    

    # ── Sort by time to show newest first ──────────────────────────────────────
    date_candidates = ["Transaction Datetime", "Transaction Date", "Datetime", "Date", "Waktu Transaksi", "Tanggal Transaksi", "waktu", "tanggal"]
    for d_col in date_candidates:
        actual_col = next((c for c in df_raw.columns if c.lower() == d_col.lower()), None)
        if actual_col:
            try:
                # Konversi ke datetime (bisa memakan waktu jika data sangat besar, errors='coerce' agar aman)
                df_raw[actual_col] = pd.to_datetime(df_raw[actual_col], errors='coerce')
                df_raw = df_raw.sort_values(by=actual_col, ascending=False).reset_index(drop=True)
                break
            except Exception:
                continue

    # Validasi apakah kolom pilihan ada di df_raw
    if bill_col not in df_raw.columns or name_col not in df_raw.columns:
        detected_cols = ", ".join([f"'{col}'" for col in df_raw.columns])
        raise ValueError(
            f"Kolom '{bill_col}' atau '{name_col}' tidak ditemukan di berkas.\n"
            f"Kolom yang tersedia: [{detected_cols}]"
        )

    # ── Auto-filter MENU PAKET dari df_raw_original sebelum memotong kolom ────────────────
    # Filter ini akan mengeluarkan item paket dari data asli, tetapi kami tetap menyimpan versi asli di df_raw_original untuk ditampilkan di tab 'Dataset Mentah'.
    filter_mask = pd.Series(False, index=df_raw_original.index)
    
    # 1. Cek di nama produk
    if name_col in df_raw_original.columns:
        filter_mask |= df_raw_original[name_col].astype(str).str.contains("MENU PAKET|PAKET", case=False, na=False)
    
    # 2. Cek di category_col (jika dipetakan)
    if category_col and category_col in df_raw_original.columns:
        filter_mask |= df_raw_original[category_col].astype(str).str.contains("MENU PAKET|PAKET", case=False, na=False)
    
    # 3. Cek di semua kolom yang mengandung kata 'category' atau 'kategori'
    cat_cols = [c for c in df_raw_original.columns if 'category' in c.lower() or 'kategori' in c.lower() or 'grup' in c.lower()]
    for c in cat_cols:
        filter_mask |= df_raw_original[c].astype(str).str.contains("MENU PAKET|PAKET", case=False, na=False)
    
    menu_paket_removed = int(filter_mask.sum())
    # DataFrame yang akan diproses selanjutnya (df_filtered) tanpa menu paket
    df_filtered = df_raw_original[~filter_mask]

    # ── Pilih kolom yang dibutuhkan dan rename ke standar ────────────────────────
    if category_col and category_col in df_filtered.columns:
        df_clean = df_filtered[[bill_col, name_col, category_col]].copy()
        df_clean.columns = ["Bill Number", "Product Name", "Product Category"]
    else:
        df_clean = df_filtered[[bill_col, name_col]].copy()
        df_clean.columns = ["Bill Number", "Product Name"]
        df_clean["Product Category"] = "Umum"

    # Hapus baris kosong
    df_clean = df_clean.dropna(subset=["Bill Number", "Product Name"])
    df_clean["Bill Number"] = df_clean["Bill Number"].astype(str).str.strip()
    df_clean["Product Name"] = df_clean["Product Name"].astype(str).str.strip()
    df_clean["Product Category"] = df_clean["Product Category"].astype(str).str.strip()

    # Tambah kolom tipe (Makanan / Minuman / Lainnya)
    df_clean["Tipe"] = df_clean["Product Category"].apply(classify_category)

    # Detect duplicate rows based on Bill Number and Product Name (dari df_clean)
    duplicate_mask = df_clean.duplicated(subset=["Bill Number", "Product Name"], keep=False)
    logger.info("Found %d duplicate rows in cleaned data.", duplicate_mask.sum())

    # Ambil bill number duplikat (dalam format asli / sebelum strip), lalu cari di df_raw_original
    # Ini agar bill number di "Contoh Data Duplikat" cocok dengan apa yang terlihat di tab Dataset Mentah
    dup_bill_numbers_clean = set(df_clean.loc[duplicate_mask, "Bill Number"].tolist())
    # Cari di df_raw_original menggunakan kolom bill asli (dengan strip agar matching)
    raw_bill_stripped = df_raw_original[bill_col].astype(str).str.strip()
    raw_dup_mask = raw_bill_stripped.isin(dup_bill_numbers_clean)
    duplicates_df = df_raw_original[raw_dup_mask].copy()

    # ── Basket (per transaksi → list produk) ──────────────────────────────────
    basket = (
        df_clean.groupby("Bill Number", sort=False)["Product Name"]
        .apply(list)
    )

    # ── Basket matrix (one-hot encoding) untuk FP-Growth ──────────────────────
    # Hanya transaksi dengan ≥ 2 item yang relevan untuk association rules
    basket_multi = basket[basket.apply(len) >= 2]

    from mlxtend.preprocessing import TransactionEncoder

    te = TransactionEncoder()
    te_array = te.fit_transform(basket_multi.tolist())
    basket_matrix = pd.DataFrame(te_array, columns=te.columns_)

    # ── Statistik ──────────────────────────────────────────────────────────────
    stats = {
        "total_baris_raw": initial_raw_count,
        "total_baris_bersih": len(df_clean),
        "total_transaksi": len(basket),
        "total_transaksi_multi": len(basket_multi),
        "total_produk_unik": df_clean["Product Name"].nunique(),
        "missing_product_name": int(df_raw_original[name_col].isna().sum()) if name_col in df_raw_original.columns else 0,
        "missing_bill_number": int(df_raw_original[bill_col].isna().sum()) if bill_col in df_raw_original.columns else 0,
        "duplicate_rows": int(duplicate_mask.sum()),
        "menu_paket_removed": menu_paket_removed,
    }

    # ── Distribusi kategori ────────────────────────────────────────────────────
    category_dist = (
        df_clean.groupby("Product Category")["Product Name"]
        .count()
        .sort_values(ascending=False)
    )

    # ── Distribusi Makanan / Minuman / Lainnya ────────────────────────────────
    type_dist = df_clean["Tipe"].value_counts()

    # ── Produk terlaris ────────────────────────────────────────────────────────
    top_products = (
        df_clean["Product Name"]
        .value_counts()
        .head(15)
    )

    return {
        "df_raw": df_filtered,
        "df_raw_original": df_raw_original,
        "df_clean": df_clean,
        "basket": basket,
        "basket_matrix": basket_matrix,
        "stats": stats,
        "category_dist": category_dist,
        "type_dist": type_dist,
        "top_products": top_products,
        "duplicates": duplicates_df,
    }