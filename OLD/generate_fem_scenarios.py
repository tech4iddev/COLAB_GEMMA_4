import json
import os
import random

def generate_scenarios(output_jsonl):
    scenarios = []
    
    # 1. Variasi Skenario Balok (Beams)
    beam_lengths = [3, 4, 5, 6, 8, 10]
    loads = [5, 10, 15, 20, 25] # kN/m
    
    for L in beam_lengths:
        for q in loads:
            m_max = (q * L**2) / 8
            scenarios.append({
                "instruction": f"Skenario Analisa Struktur: Balok sederhana (Simple Beam) bentang {L} meter dengan beban merata (UDL) sebesar {q} kN/m.",
                "input": "Berapa Momen Maksimum (Mmax) dan tuliskan kode OpenSeesPy nya?",
                "output": f"Hasil Analisa:\n- Momen Maksimum: {m_max:.2f} kNm (berada di tengah bentang)\n- Kode OpenSeesPy:\n```python\nops.node(1, 0, 0); ops.node(2, {L}, 0)\nops.fix(1, 1, 1, 0); ops.fix(2, 0, 1, 0)\nops.timeSeries('Linear', 1)\nops.pattern('Plain', 1, 1)\nops.eleLoad('-ele', 1, '-type', '-beamUniform', -{q})\n```"
            })

    # 2. Variasi Skenario Cantilever
    for L in [2, 3, 4]:
        for P in [10, 20, 30]:
            m_cant = P * L
            scenarios.append({
                "instruction": f"Skenario Analisa Struktur: Balok Kantilever panjang {L} meter dengan beban titik (Point Load) {P} kN di ujung bebas.",
                "input": "Analisa momen di tumpuan.",
                "output": f"Momen pada tumpuan jepit adalah {m_cant:.2f} kNm. Penampang harus didesain untuk menahan momen negatif pada bagian atas balok."
            })

    # 3. Skenario Portal (Frame) Lateral
    heights = [3.5, 4, 7] # m
    wind_loads = [2, 5, 8] # kN (lateral)
    for H in heights:
        for F in wind_loads:
            scenarios.append({
                "instruction": f"Skenario Portal Bangunan 2D: Tinggi kolom {H} meter dengan beban lateral angin {F} kN di puncak.",
                "input": "Bagaimana distribusi gaya dalam secara umum?",
                "output": f"Pada portal dengan beban lateral {F} kN dan tinggi {H} m:\n1. Kolom mengalami momen sebesar {F*H:.2f} kNm pada dasar tumpuan (jika jepit).\n2. Terjadi gaya geser (shear) konstan pada seluruh tinggi kolom.\n3. Balok akan mengalami gaya aksial akibat transfer beban lateral."
            })

    # 4. Skenario Matriks Kekakuan
    e_vals = [200000, 210000, 200] # MPa/GPa
    for E in e_vals:
        scenarios.append({
            "instruction": f"Bagaimana pengaruh modulus elastisitas E = {E} terhadap kekakuan elemen?",
            "input": "",
            "output": f"Kekakuan elemen (k) berbanding lurus dengan Modulus Elastisitas (E). Semakin tinggi E ({E}), maka matriks kekakuan elemen [k = AE/L] akan semakin besar, yang mengakibatkan perpindahan (displacement) struktur menjadi semakin kecil di bawah beban yang sama."
        })

    # Simpan
    if not os.path.exists("training_data"):
        os.makedirs("training_data")
        
    with open(output_jsonl, "w") as f:
        for entry in scenarios:
            f.write(json.dumps(entry) + "\n")
            
    print(f"✅ Berhasil membuat {len(scenarios)} Skenario FEM di {output_jsonl}")

if __name__ == "__main__":
    output_path = "/Users/caturimamsaputro/Documents/GEMMA_4_unsloth/training_data/dataset_fem_scenarios.jsonl"
    generate_scenarios(output_path)
