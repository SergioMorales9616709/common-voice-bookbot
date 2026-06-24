from mls_metadata import download_speaker_metadata, get_female_speakers

print("\n--- Descargando metadatos de hablantes MLS Spanish ---")
df = download_speaker_metadata()

females = get_female_speakers(df)
top = females.head(10).reset_index(drop=True)

print("\nTop hablantes femeninas en MLS Spanish (por minutos de audio):\n")
print(f"{'#':<4} {'speaker_id':<14} {'minutos':>10}")
print("-" * 32)
for i, row in top.iterrows():
    print(f"{i + 1:<4} {row['speaker_id']:<14} {row['minutes']:>10.1f}")

best = top.iloc[0]["speaker_id"]
print(f"\nPara exportar la hablante con más audio:")
print(f"  uv run exportar_dataset.py --speaker {best}")
