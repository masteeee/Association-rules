# pyrefly: ignore [missing-import]
# pyrefly: ignore [missing-import]
from flask import Flask, render_template, request, jsonify
import os
import pandas as pd

from services.preprocessing import preprocess
from services.fpgrowth_service import run_fpgrowth
from services.apriori_service import run_apriori

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    # Cek file
    if "file" not in request.files:
        return "Tidak ada file yang dipilih"

    file = request.files["file"]

    if file.filename == "":
        return "File belum dipilih"

    # Simpan file dengan nama unik
    import uuid
    ext = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        unique_filename
    )
    file.save(filepath)

    # Baca header file untuk pemetaan kolom
    try:
        if filepath.lower().endswith(".csv"):
            try:
                # Coba deteksi separator secara otomatis (misal koma atau titik koma)
                df_temp = pd.read_csv(filepath, sep=None, engine="python", nrows=5, encoding="utf-8")
            except Exception:
                # Fallback jika gagal
                df_temp = pd.read_csv(filepath, nrows=5, encoding="utf-8")
        else:
            df_temp = pd.read_excel(filepath, nrows=5)
    except Exception as e:
        return f"Gagal membaca berkas: {str(e)}"

    columns = df_temp.columns.str.strip().tolist()

    # Coba auto-detect kolom untuk default selection
    def normalize(s):
        return "".join(c for c in str(s).lower() if c.isalnum())

    normalized_cols = {normalize(c): c for c in columns}

    bill_candidates = ["billnumber", "billno", "nobill", "nonota", "nomornota", "nota", "transactionid", "transactionno", "idtransaksi"]
    name_candidates = ["productname", "namaproduk", "namabarang", "itemname", "nama", "item"]
    category_candidates = ["productcategory", "kategoriproduk", "kategoribarang", "kategori", "category", "productgroup", "grup"]

    detected_bill = next((normalized_cols[cand] for cand in bill_candidates if cand in normalized_cols), columns[0] if columns else "")
    detected_name = next((normalized_cols[cand] for cand in name_candidates if cand in normalized_cols), columns[1] if len(columns) > 1 else "")
    detected_category = next((normalized_cols[cand] for cand in category_candidates if cand in normalized_cols), "")

    # Ambil 5 baris pertama untuk preview
    preview_data = df_temp.head(5).to_dict(orient="records")

    return render_template(
        "select_columns.html",
        filename=file.filename,
        unique_filename=unique_filename,
        columns=columns,
        detected_bill=detected_bill,
        detected_name=detected_name,
        detected_category=detected_category,
        preview_data=preview_data
    )


@app.route("/process", methods=["POST"])
def process():
    unique_filename = request.form.get("unique_filename")
    original_filename = request.form.get("original_filename", "dataset")

    if not unique_filename:
        return "Berkas tidak ditemukan."

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
    if not os.path.exists(filepath):
        return "Berkas tidak ditemukan pada server."

    # Ambil pemetaan kolom dari form
    bill_col = request.form.get("bill_col")
    name_col = request.form.get("name_col")
    category_col = request.form.get("category_col")

    if not bill_col or not name_col:
        return "Kolom ID Transaksi dan Nama Produk wajib diisi."

    if category_col == "none":
        category_col = None

    # Ambil parameter min_support dan min_confidence (dalam persen, e.g. 2 dan 30)
    try:
        min_support_pct = int(float(request.form.get("min_support", 2)))
    except ValueError:
        min_support_pct = 2
    min_support = min_support_pct / 100.0

    try:
        min_confidence_pct = int(float(request.form.get("min_confidence", 30)))
    except ValueError:
        min_confidence_pct = 30
    min_confidence = min_confidence_pct / 100.0

    try:
        # Preprocessing kustom dengan pemetaan kolom dari user
        prep = preprocess(filepath, bill_col, name_col, category_col)
    except Exception as e:
        return f"Error saat melakukan preprocessing data: {str(e)}"

    # Ambil data raw dan stats
    df_raw_original = prep["df_raw_original"]
    df_clean = prep["df_clean"]
    basket = prep["basket"]
    basket_matrix = prep["basket_matrix"]
    stats = prep["stats"]

    # Previews — hanya 100 baris awal dikirim ke template (data lengkap via AJAX)
    INITIAL_ROWS = 100

    # Raw preview
    raw_columns = df_raw_original.columns.tolist()
    raw_preview = df_raw_original.head(INITIAL_ROWS).to_dict(orient="records")
    raw_total = len(df_raw_original)

    # Clean preview
    display_cols = ["Bill Number", "Product Name"]
    rename_map = {"Bill Number": bill_col, "Product Name": name_col}
    if category_col:
        display_cols.append("Product Category")
        rename_map["Product Category"] = category_col
    df_clean_display = df_clean[display_cols].rename(columns=rename_map)
    if "Tipe" in df_clean.columns:
        df_clean_display["Tipe"] = df_clean["Tipe"].values
    clean_columns = df_clean_display.columns.tolist()
    clean_preview = df_clean_display.head(INITIAL_ROWS).to_dict(orient="records")
    clean_total = len(df_clean_display)

    # Basket preview dengan nama kolom sesuai filter user
    basket_df = basket.reset_index()
    rename_basket_dict = {"Bill Number": bill_col, "Product Name": name_col}
    basket_df_preview = basket_df.rename(columns=rename_basket_dict)
    basket_columns = basket_df_preview.columns.tolist()
    basket_rows_initial = basket_df_preview.head(INITIAL_ROWS).to_dict(orient="records")
    basket_total = len(basket_df_preview)
    # Ubah list produk jadi string koma untuk preview
    for row in basket_rows_initial:
        row[name_col] = ", ".join(row[name_col])
    basket_preview = basket_rows_initial

    # Jalankan algoritma
    fp_results = run_fpgrowth(basket_matrix, min_support=min_support, min_confidence=min_confidence)
    ap_results = run_apriori(basket_matrix, min_support=min_support, min_confidence=min_confidence)

    # Format data untuk visualisasi
    top_products_names = prep["top_products"].index.tolist()
    top_products_counts = prep["top_products"].values.tolist()

    type_dist_labels = prep["type_dist"].index.tolist()
    type_dist_values = prep["type_dist"].values.tolist()

    category_dist_labels = prep["category_dist"].index.tolist()
    category_dist_values = prep["category_dist"].values.tolist()

    # Model evaluation comparison
    comparison_metrics = {
        "apriori_time": ap_results.get("elapsed_time", 0.0),
        "fpgrowth_time": fp_results.get("elapsed_time", 0.0),
        "apriori_rules_count": len(ap_results["rules"]) if not ap_results["rules"].empty else 0,
        "fpgrowth_rules_count": len(fp_results["rules"]) if not fp_results["rules"].empty else 0,
        "apriori_itemsets_count": len(ap_results["frequent_itemsets"]) if not ap_results["frequent_itemsets"].empty else 0,
        "fpgrowth_itemsets_count": len(fp_results["frequent_itemsets"]) if not fp_results["frequent_itemsets"].empty else 0,
    }

    # Urutan rules ke list of dict
    fp_rules = fp_results["rules"].to_dict(orient="records") if not fp_results["rules"].empty else []
    ap_rules = ap_results["rules"].to_dict(orient="records") if not ap_results["rules"].empty else []

    fp_itemsets = fp_results["frequent_itemsets"].to_dict(orient="records") if not fp_results["frequent_itemsets"].empty else []
    ap_itemsets = ap_results["frequent_itemsets"].to_dict(orient="records") if not ap_results["frequent_itemsets"].empty else []

    # Ambil promo recommendations dari rules
    # Kita buat rekomendasi gabungan dari FP-Growth (kedua metode harusnya sama hasilnya)
    promo_rules = []
    if not fp_results["rules"].empty:
        # filter rules dengan lift > 1 dan confidence tinggi
        filtered_rules = fp_results["rules"][fp_results["rules"]["lift"] >= 1.0].sort_values("lift", ascending=False)
        # ambil top 10 rules
        promo_rules = filtered_rules.head(10).to_dict(orient="records")
    elif not ap_results["rules"].empty:
        filtered_rules = ap_results["rules"][ap_results["rules"]["lift"] >= 1.0].sort_values("lift", ascending=False)
        promo_rules = filtered_rules.head(10).to_dict(orient="records")

    return render_template(
        "result.html",
        filename=original_filename,
        min_support=min_support_pct,
        min_confidence=min_confidence_pct,
        stats=stats,
        raw_columns=raw_columns,
        raw_preview=raw_preview,
        raw_total=raw_total,
        clean_columns=clean_columns,
        clean_preview=clean_preview,
        clean_total=clean_total,
        basket_columns=basket_columns,
        basket_preview=basket_preview,
        basket_total=basket_total,
        duplicates_preview=prep["duplicates"].head(500).to_dict(orient="records") if not prep["duplicates"].empty else [],
        fp_error=fp_results["error"],
        ap_error=ap_results["error"],
        comparison_metrics=comparison_metrics,
        fp_rules=fp_rules,
        ap_rules=ap_rules,
        fp_itemsets=fp_itemsets,
        ap_itemsets=ap_itemsets,
        top_products_names=top_products_names,
        top_products_counts=top_products_counts,
        type_dist_labels=type_dist_labels,
        type_dist_values=type_dist_values,
        category_dist_labels=category_dist_labels,
        category_dist_values=category_dist_values,
        promo_rules=promo_rules,
        unique_filename=unique_filename,
        bill_col=bill_col,
        name_col=name_col,
        category_col=category_col or ""
    )


def sanitize_nan(val):
    if isinstance(val, dict):
        return {k: sanitize_nan(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [sanitize_nan(v) for v in val]
    elif isinstance(val, float):
        import math
        if math.isnan(val) or math.isinf(val):
            return None
    elif pd.isna(val):
        return None
    elif hasattr(val, "isoformat"):
        return val.isoformat()
    return val


@app.route("/get_preview_data", methods=["POST"])
def get_preview_data():
    """Endpoint untuk lazy-loading data tabel preview (raw/clean/basket) via AJAX."""
    data = request.get_json()
    unique_filename = data.get("unique_filename", "")
    bill_col = data.get("bill_col", "")
    name_col = data.get("name_col", "")
    category_col = data.get("category_col") or None
    table_type = data.get("table_type", "raw")  # raw | clean | basket
    page = int(data.get("page", 1))
    page_size = int(data.get("page_size", 200))

    if not unique_filename or not bill_col or not name_col:
        return jsonify({"error": "Parameter tidak lengkap."}), 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File tidak ditemukan."}), 404

    try:
        prep = preprocess(filepath, bill_col, name_col, category_col)

        start = (page - 1) * page_size
        end = start + page_size

        if table_type == "raw":
            df = prep["df_raw_original"]
            columns = df.columns.tolist()
            rows = df.iloc[start:end].to_dict(orient="records")
            total = len(df)

        elif table_type == "clean":
            df_clean = prep["df_clean"]
            display_cols = ["Bill Number", "Product Name"]
            rename_map = {"Bill Number": bill_col, "Product Name": name_col}
            if category_col:
                display_cols.append("Product Category")
                rename_map["Product Category"] = category_col
            df = df_clean[display_cols].rename(columns=rename_map)
            if "Tipe" in df_clean.columns:
                df["Tipe"] = df_clean["Tipe"].values
            columns = df.columns.tolist()
            rows = df.iloc[start:end].to_dict(orient="records")
            total = len(df)

        elif table_type == "basket":
            basket = prep["basket"]
            basket_df = basket.reset_index().rename(
                columns={"Bill Number": bill_col, "Product Name": name_col}
            )
            columns = basket_df.columns.tolist()
            slice_df = basket_df.iloc[start:end].copy()
            # Konversi list produk ke string koma
            slice_df[name_col] = slice_df[name_col].apply(
                lambda x: ", ".join(x) if isinstance(x, list) else str(x)
            )
            rows = slice_df.to_dict(orient="records")
            total = len(basket_df)

        else:
            return jsonify({"error": "table_type tidak dikenali."}), 400

        return jsonify({
            "columns": columns,
            "rows": sanitize_nan(rows),
            "total": total,
            "page": page,
            "page_size": page_size,
            "has_more": end < total
        })

    except Exception as e:
        return jsonify({"error": f"Gagal memuat data: {str(e)}"}), 500


@app.route("/get_rule_transactions", methods=["POST"])
def get_rule_transactions():
    """Mengembalikan daftar transaksi (basket) yang memenuhi aturan asosiasi tertentu."""
    data = request.get_json()
    unique_filename = data.get("unique_filename", "")
    antecedents_str = data.get("antecedents", "")
    consequents_str = data.get("consequents", "")
    bill_col = data.get("bill_col", "")
    name_col = data.get("name_col", "")
    category_col = data.get("category_col") or None

    if not unique_filename or not bill_col or not name_col:
        return jsonify({"error": "Parameter tidak lengkap."}), 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File dataset tidak ditemukan di server."}), 404

    try:
        prep = preprocess(filepath, bill_col, name_col, category_col)
        basket = prep["basket"]  # Series: index=Bill Number (stripped), value=list of products
        df_raw_original = prep["df_raw_original"]

        # Buat peta: bill number bersih → nilai asli di file (sebelum strip)
        # Ini agar user bisa mencari bill number di Dataset Mentah (kolom asli)
        raw_bill_series = df_raw_original[bill_col].astype(str)
        clean_to_raw = {}
        for raw_val in raw_bill_series:
            clean_val = raw_val.strip()
            if clean_val not in clean_to_raw:
                clean_to_raw[clean_val] = raw_val  # ambil nilai asli pertama yang cocok

        # Parse item sets dari string (dipisahkan koma)
        antecedents = set(item.strip() for item in antecedents_str.split(",") if item.strip())
        consequents = set(item.strip() for item in consequents_str.split(",") if item.strip())
        rule_items = antecedents | consequents

        # Filter transaksi yang mengandung SEMUA item antecedents DAN consequents
        def matches_rule(product_list):
            product_set = set(product_list)
            return antecedents.issubset(product_set) and consequents.issubset(product_set)

        matched = basket[basket.apply(matches_rule)]

        # Bangun response: list of {bill_number, raw_bill_number, products, total_items}
        transactions = []
        for bill_number, products in matched.items():
            clean_bill = str(bill_number)
            raw_bill = clean_to_raw.get(clean_bill, clean_bill)  # fallback ke clean jika tidak ditemukan
            transactions.append({
                "bill_number": clean_bill,
                "raw_bill_number": raw_bill,  # nilai asli di file untuk pencarian di Dataset Mentah
                "products": sorted(set(products)),  # deduplicate & sort
                "total_items": len(set(products))
            })

        return jsonify({
            "transactions": transactions,
            "total": len(transactions),
            "antecedents": list(antecedents),
            "consequents": list(consequents)
        })

    except Exception as e:
        return jsonify({"error": f"Gagal memuat transaksi: {str(e)}"}), 500

@app.route("/search_by_bill", methods=["POST"])
def search_by_bill():
    """Search for a specific bill number across raw, cleaned, and basket data.
    Enables lookup in the Preprocessing & Preview tab without heavy queries.
    """
    data = request.get_json()
    unique_filename = data.get("unique_filename", "")
    bill_col = data.get("bill_col", "")
    name_col = data.get("name_col", "")
    category_col = data.get("category_col") or None
    target_bill = data.get("bill_number", "").strip()

    if not unique_filename or not bill_col or not name_col or not target_bill:
        return jsonify({"error": "Parameter tidak lengkap."}), 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File tidak ditemukan."}), 404

    try:
        prep = preprocess(filepath, bill_col, name_col, category_col)
        df_raw_original = prep["df_raw_original"]
        df_clean = prep["df_clean"]
        basket = prep["basket"]

        # Raw rows (original value may include whitespace)
        raw_match = df_raw_original[df_raw_original[bill_col].astype(str).str.strip() == target_bill]
        raw_rows = raw_match.to_dict(orient="records") if not raw_match.empty else []

        # Clean rows (already stripped)
        clean_match = df_clean[df_clean["Bill Number"].astype(str).str.strip() == target_bill]
        
        clean_rows = []
        if not clean_match.empty:
            display_cols = ["Bill Number", "Product Name"]
            rename_map = {"Bill Number": bill_col, "Product Name": name_col}
            if category_col:
                display_cols.append("Product Category")
                rename_map["Product Category"] = category_col
                
            df_clean_display = clean_match[display_cols].rename(columns=rename_map)
            if "Tipe" in clean_match.columns:
                df_clean_display["Tipe"] = clean_match["Tipe"].values
            clean_rows = df_clean_display.to_dict(orient="records")

        # Basket items for this bill
        basket_items = basket.get(target_bill, [])
        if isinstance(basket_items, list):
            basket_items = sorted(list(set(basket_items)))
        elif basket_items is not None:
            # handle if it's a pandas series or array or string
            basket_items = sorted(list(set(list(basket_items))))
        else:
            basket_items = []

        return jsonify({
            "raw_rows": sanitize_nan(raw_rows),
            "clean_rows": sanitize_nan(clean_rows),
            "basket_items": sanitize_nan(basket_items),
        })
    except Exception as e:
        return jsonify({"error": f"Gagal mencari bill: {str(e)}"}), 500



if __name__ == "__main__":
    host = "0.0.0.0"
    port = int(os.environ.get("PORT", 5000))
    app.run(host=host, port=port, debug=True)