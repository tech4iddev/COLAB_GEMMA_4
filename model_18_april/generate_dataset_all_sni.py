import json
import random
import math
import hashlib
import os

seen = set()

# =========================
# UTIL
# =========================
def unique(text):
    h = hashlib.md5(text.encode()).hexdigest()
    if h in seen:
        return False
    seen.add(h)
    return True

def wrap(instruction, input_text, output):
    """
    Standard format for Gemma Fine-tuning.
    """
    # Jika output berupa dict, ubah jadi string terformat agar model belajar narasi
    if isinstance(output, dict):
        output_str = "Hasil Analisa:\n" + "\n".join([f"- {k}: {v}" for k, v in output.items()])
    else:
        output_str = str(output)
        
    return {
        "instruction": instruction.strip(),
        "input": input_text.strip(),
        "output": output_str.strip()
    }

# =========================
# 1. DESIGN BETON 🔥
# =========================
def design_beam():
    Mu = random.uniform(40, 250)  # kNm
    fy = 400
    d = random.uniform(400, 600)
    phi = 0.9

    # Rumus pendekatan As = Mu / (phi * fy * 0.9 * d)
    As = (Mu * 1e6) / (phi * fy * 0.9 * d)

    instruction = "Hitung kebutuhan luas tulangan tarik (As) balok beton."
    input_text = f"Mu = {round(Mu,1)} kNm, fy = {fy} MPa, d = {round(d)} mm."
    
    output = {
        "Momen Ultimate (Mu)": f"{round(Mu,1)} kNm",
        "Luas Tulangan (As)": f"{round(As,2)} mm2",
        "Catatan": "Gunakan tulangan yang identitas luasnya mendekati atau lebih besar dari nilai As tersebut."
    }

    return wrap(instruction, input_text, output)

# =========================
# 2. LOAD COMBINATION
# =========================
def load_combo():
    D = random.uniform(2, 12)
    L = random.uniform(1, 8)
    U = 1.2 * D + 1.6 * L

    instruction = "Tentukan kombinasi beban ultimate (U) sesuai SNI 1727."
    input_text = f"Beban Mati (D) = {round(D,1)} kN/m, Beban Hidup (L) = {round(L,1)} kN/m."
    
    return wrap(instruction, input_text, {"Beban Ultimate (U)": f"{round(U,2)} kN/m"})

# =========================
# 3. GEMPA (SEISMIC)
# =========================
def seismic():
    W = random.randint(500, 5000)
    Cs = random.uniform(0.04, 0.3)
    V = Cs * W

    instruction = "Hitung Gaya Geser Dasar (Base Shear) beban gempa."
    input_text = f"Berat Seismik Efektif (W) = {W} kN, Koefisien Respons Seismik (Cs) = {round(Cs,3)}."
    
    return wrap(instruction, input_text, {"Gaya Geser Dasar (V)": f"{round(V,2)} kN"})

# =========================
# 4. BAJA (STEEL)
# =========================
def steel():
    fy = random.choice([240, 250, 345])
    A = random.randint(1200, 15000)
    Pn = fy * A / 1000
    phi_Pn = 0.9 * Pn

    instruction = "Hitung kapasitas tarik nominal dan desain batang baja."
    input_text = f"Mutu baja fy = {fy} MPa, Luas penampang A = {A} mm2."
    
    output = {
        "Kuat Tarik Nominal (Pn)": f"{round(Pn,2)} kN",
        "Kuat Tarik Desain (phi Pn)": f"{round(phi_Pn,2)} kN"
    }
    return wrap(instruction, input_text, output)

# =========================
# 5. FEM GENERATE
# =========================
def fem():
    L = random.uniform(4, 12)
    H = random.uniform(3, 4.5)

    instruction = "Buat struktur koordinat untuk model Portal 2D."
    input_text = f"Portal dengan bentang {round(L,1)} meter dan tinggi {round(H,1)} meter."

    output = {
        "Node 1": "(0, 0) - Support",
        "Node 2": f"({round(L,1)}, 0) - Support",
        "Node 3": f"({round(L,1)}, {round(H,1)}) - Joint",
        "Node 4": f"(0, {round(H,1)}) - Joint",
        "Elemen 1": "Node 1 ke 4 (Kolom)",
        "Elemen 2": "Node 4 ke 3 (Balok)",
        "Elemen 3": "Node 3 ke 2 (Kolom)"
    }
    return wrap(instruction, input_text, output)

# =========================
# 6. VALIDATION
# =========================
def validate():
    options = [
        ("1.2D + 1.6L", True),
        ("1.4D", True),
        ("1.0D + 1.0L", False),
        ("1.2D + 0.5L + 1.0E", True)
    ]
    combo, is_valid = random.choice(options)

    instruction = "Apakah kombinasi pembebanan berikut sesuai dengan SNI 1727?"
    input_text = f"Kombinasi: {combo}"
    
    msg = "Sesuai dengan standar SNI." if is_valid else "Tidak sesuai. Kombinasi ini biasanya untuk pengecekan defleksi (izin), bukan kekuatan ultimate."
    return wrap(instruction, input_text, {"Status": msg, "Valid": is_valid})

# =========================
# 7. ERROR REPAIR
# =========================
def repair():
    instruction = "Perbaiki kesalahan pada file input model struktur berikut."
    input_text = "ERROR: node_i=1, node_j=1 (Elemen 1)"
    
    output = "Kesalahan terdeteksi: Node awal dan akhir elemen tidak boleh sama. Perbaikan: Pastikan node_j merujuk pada ID node yang berbeda, misalnya node_j=2."
    return wrap(instruction, input_text, output)

# =========================
# MAIN
# =========================
def generate(n=5000, output_dir="datasets", filename="ultimate_dataset.jsonl"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Created directory: {output_dir}")

    filepath = os.path.join(output_dir, filename)
    
    funcs = [
        design_beam,
        load_combo,
        seismic,
        steel,
        fem,
        validate,
        repair
    ]

    print(f"🚀 Memulai generate {n} data ke {filepath}...")
    
    with open(filepath, "w", encoding="utf-8") as f:
        count = 0
        attempts = 0
        while count < n and attempts < n * 5:
            attempts += 1
            item = random.choice(funcs)()
            # Buat string unik berdasarkan instruksi + input untuk menghindari duplikasi
            unique_key = item["instruction"] + item["input"]
            
            if unique(unique_key):
                f.write(json.dumps(item) + "\n")
                count += 1
                if count % 1000 == 0:
                    print(f"   ✅ {count} data generated...")

    print(f"\n✨ SELESAI! {count} dataset unik telah tersimpan di {filepath}")

if __name__ == "__main__":
    # Kita generate 5000 dulu agar tidak terlalu besar tapi cukup beragam
    generate(n=5000)
