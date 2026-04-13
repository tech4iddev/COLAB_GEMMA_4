import json
import os

def generate_fem_dataset(output_jsonl):
    dataset = [
        {
            "instruction": "Bagaimana langkah dasar melakukan analisa statis 2D Truss menggunakan OpenSeesPy?",
            "input": "",
            "output": "Langkah dasar analisa 2D Truss di OpenSeesPy:\n1. Init model: `model('basic', '-ndm', 2, '-ndf', 2)`\n2. Define Nodes: `node(tag, x, y)`\n3. Define Material: `uniaxialMaterial('Elastic', tag, E)`\n4. Define Elements: `element('truss', tag, node1, node2, A, matTag)`\n5. Boundary Conditions: `fix(nodeTag, 1, 1)`\n6. Loads: `load(nodeTag, Fx, Fy)`\n7. Analysis: `analyze(1)`"
        },
        {
            "instruction": "Apa fungsi matriks kekakuan (stiffness matrix) dalam metode elemen hingga?",
            "input": "",
            "output": "Matriks kekakuan (k) merepresentasikan hubungan antara gaya (P) dan perpindahan (u) pada suatu elemen, sesuai hukum Hooke: P = k.u. Dalam FEM, matriks kekakuan elemen digabungkan menjadi matriks kekakuan global (K) untuk menyelesaikan perpindahan di seluruh titik join (node) struktur."
        },
        {
            "instruction": "Tuliskan contoh kode OpenSeesPy untuk balok sederhana (Simple Beam).",
            "input": "",
            "output": "import openseespy.opensees as ops\nops.model('basic', '-ndm', 2, '-ndf', 3)\nops.node(1, 0, 0); ops.node(2, 5, 0)\nops.fix(1, 1, 1, 0); ops.fix(2, 0, 1, 0)\nops.geomTransf('Linear', 1)\nops.element('elasticBeamColumn', 1, 1, 2, A, E, Iz, 1)\nops.timeSeries('Constant', 1)\nops.pattern('Plain', 1, 1)\nops.load(2, 0, -10, 0)\nops.analysis('Static')\nops.analyze(1)\nprint('Analisa Selesai')"
        }
    ]

    # Simpan ke folder training_data
    if not os.path.exists("training_data"):
        os.makedirs("training_data")

    with open(output_jsonl, "w") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")
            
    print(f"✅ Berhasil membuat data FEM & OpenSeesPy di {output_jsonl}")

if __name__ == "__main__":
    output_file = "/Users/caturimamsaputro/Documents/GEMMA_4_unsloth/training_data/dataset_fem_opensees.jsonl"
    generate_fem_dataset(output_file)
