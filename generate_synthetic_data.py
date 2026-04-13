import json
import random
import math
import os

def generate_concrete_flexure(num_samples=500):
    dataset = []
    print(f"Generating {num_samples} samples...")
    for _ in range(num_samples):
        # 1. Randomize Parameters
        b = random.choice([200, 250, 300, 350, 400])
        d = random.choice([400, 450, 500, 600])
        fc = random.choice([20, 25, 30, 35])
        fy = random.choice([280, 420])
        
        n_bars = random.randint(2, 6)
        d_bar = random.choice([16, 19, 22, 25])
        as_total = n_bars * (0.25 * math.pi * d_bar**2)
        
        # 2. Calculation Logic
        beton_085 = 0.85
        a = (as_total * fy) / (beton_085 * fc * b)
        mn = (as_total * fy * (d - a/2)) / 10**6 # kNm
        
        # 3. Instruction format
        instruction = "Hitung kapasitas momen nominal (Mn) balok beton bertulang persegi sesuai SNI 2847:2019."
        input_text = f"Lebar b={b}mm, Tinggi efektif d={d}mm, Tulangan As={n_bars}D{d_bar} ({as_total:.1f} mm2), f'c={fc}MPa, fy={fy}MPa."
        
        output_text = (
            f"Berdasarkan SNI 2847:2019:\n"
            f"1. Kedalaman blok tekan persegi (a) = (As * fy) / (0.85 * f'c * b)\n"
            f"   a = ({as_total:.1f} * {fy}) / (0.85 * {fc} * {b}) = {a:.2f} mm.\n"
            f"2. Kapasitas Momen Nominal (Mn) = As * fy * (d - a/2)\n"
            f"   Mn = {as_total:.1f} * {fy} * ({d} - {a:.2f}/2) / 10^6 = {mn:.2f} kNm.\n"
            f"Jadi, kapasitas momen nominal balok adalah {mn:.2f} kNm."
        )
        
        dataset.append({
            "instruction": instruction,
            "input": input_text,
            "output": output_text
        })
    return dataset

if __name__ == "__main__":
    output_file = "dataset_synthetic_concrete.jsonl"
    data = generate_concrete_flexure(500)
    with open(output_file, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
    print(f"✅ Created {output_file}")
