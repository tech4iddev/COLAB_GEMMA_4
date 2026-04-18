import json
import random
import math
import os

def generate_steel_capacity(num_samples=300):
    """
    Generator data sintetis untuk kapasitas aksial baja (SNI 1729:2020).
    """
    dataset = []
    for _ in range(num_samples):
        A_area = random.choice([2500, 3500, 5000, 8500, 12000])
        Fy = random.choice([240, 250, 290, 345])
        E = 200000
        L = random.choice([2000, 3000, 4000, 6000])
        r = random.choice([30, 45, 60, 85])
        K = 1.0
        
        langsing = (K * L) / r
        Fe = (math.pi**2 * E) / (langsing**2)
        
        if langsing <= 4.71 * math.sqrt(E/Fy):
            Fcr = (0.658**(Fy/Fe)) * Fy
        else:
            Fcr = 0.877 * Fe
            
        Pn = (Fcr * A_area) / 1000
        phi_Pn = 0.9 * Pn
        
        instruction = "Hitung kuat tekan desain (phi Pn) kolom baja berdasarkan SNI 1729:2020."
        input_text = f"Data profil: Luas penampang Ag={A_area} mm2, Jari-jari girasi r={r} mm, Fy={Fy} MPa, E={E} MPa. Panjang batang L={L} mm dengan tumpuan sendi-sendi (K=1.0)."
        
        output_text = (
            f"Analisa berdasarkan SNI 1729:2020:\n"
            f"1. Hitung Rasio Kelangsingan (λ) = {langsing:.2f}\n"
            f"2. Hitung Tegangan Tekuk Elastis (Fe) = {Fe:.2f} MPa\n"
            f"3. Hitung Tegangan Kritis (Fcr) = {Fcr:.2f} MPa\n"
            f"4. Kuat Tekan Nominal (Pn) = {Pn:.2f} kN\n"
            f"5. Kuat Tekan Desain (φPn) = {phi_Pn:.2f} kN."
        )
        
        dataset.append({
            "instruction": instruction,
            "input": input_text,
            "output": output_text
        })
    return dataset

def save_jsonl(data, filename):
    with open(filename, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
    print(f"✅ Created {filename} with {len(data)} samples.")

if __name__ == "__main__":
    steel_data = generate_steel_capacity(500)
    save_jsonl(steel_data, "dataset_analysis.jsonl")
