import pandas as pd
import time
from mlxtend.frequent_patterns import apriori, association_rules


def run_apriori(basket_matrix, min_support=0.02, min_confidence=0.3):
    """
    Menjalankan Apriori dan menghasilkan association rules.

    Parameters:
        basket_matrix  : DataFrame one-hot encoding hasil preprocessing
        min_support    : float, minimum support (default 0.02 = 2%)
        min_confidence : float, minimum confidence (default 0.3 = 30%)

    Returns:
        dict berisi:
          - frequent_itemsets : DataFrame frequent itemsets terurut support desc
          - rules             : DataFrame association rules terurut lift desc
          - error             : str pesan error jika gagal, None jika sukses
          - elapsed_time      : float, waktu eksekusi dalam detik
    """
    start_time = time.time()
    try:
        if basket_matrix.empty or len(basket_matrix.columns) < 2:
            return {
                "frequent_itemsets": pd.DataFrame(),
                "rules": pd.DataFrame(),
                "error": "Data tidak cukup untuk dianalisis. Pastikan ada transaksi dengan ≥2 produk berbeda.",
                "elapsed_time": 0.0
            }

        # ── Frequent Itemsets ─────────────────────────────────────────────────
        frequent_itemsets = apriori(
            basket_matrix,
            min_support=min_support,
            use_colnames=True
        )

        if frequent_itemsets.empty:
            return {
                "frequent_itemsets": pd.DataFrame(),
                "rules": pd.DataFrame(),
                "error": (
                    f"Tidak ada itemset yang memenuhi minimum support {min_support:.0%}. "
                    "Coba turunkan nilai minimum support."
                ),
                "elapsed_time": time.time() - start_time
            }

        # Tambah kolom panjang itemset & jumlah (count)
        frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(len)
        frequent_itemsets["itemsets_str"] = frequent_itemsets["itemsets"].apply(
            lambda x: ", ".join(sorted(x))
        )
        frequent_itemsets = frequent_itemsets.sort_values("support", ascending=False)

        # ── Association Rules ─────────────────────────────────────────────────
        rules = association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=min_confidence,
            num_itemsets=len(frequent_itemsets)
        )

        if rules.empty:
            return {
                "frequent_itemsets": frequent_itemsets,
                "rules": pd.DataFrame(),
                "error": (
                    f"Tidak ada rules yang memenuhi minimum confidence {min_confidence:.0%}. "
                    "Coba turunkan nilai minimum confidence."
                ),
                "elapsed_time": time.time() - start_time
            }

        # Format kolom tampilan
        rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
        rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))

        # Urutkan berdasarkan lift DESC, lalu confidence DESC
        rules = rules.sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)

        # Kolom rekomendasi promosi
        def promo_label(row):
            if row["lift"] >= 2 and row["confidence"] >= 0.5:
                return "🔥 Sangat Direkomendasikan"
            elif row["lift"] >= 1.5:
                return "✅ Direkomendasikan"
            elif row["lift"] >= 1.0:
                return "💡 Pertimbangkan"
            else:
                return "—"

        rules["rekomendasi"] = rules.apply(promo_label, axis=1)
        elapsed_time = time.time() - start_time

        return {
            "frequent_itemsets": frequent_itemsets,
            "rules": rules,
            "error": None,
            "elapsed_time": elapsed_time
        }

    except Exception as e:
        return {
            "frequent_itemsets": pd.DataFrame(),
            "rules": pd.DataFrame(),
            "error": f"Error saat menjalankan Apriori: {str(e)}",
            "elapsed_time": time.time() - start_time
        }