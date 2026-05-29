from PIL import Image
from pathlib import Path

alice_dir = Path("alice")
bob_dir = Path("bob")
out_dir = Path("paired")
out_dir.mkdir(exist_ok=True)

for i in range(1, 17):
    alice_path = alice_dir / f"video_{i:02d}_alice.png"
    bob_path = bob_dir / f"video_{i:02d}_bob.png"

    alice_img = Image.open(alice_path).convert("RGB")
    bob_img = Image.open(bob_path).convert("RGB")

    # Make both images the same height
    h = min(alice_img.height, bob_img.height)

    alice_w = int(alice_img.width * h / alice_img.height)
    bob_w = int(bob_img.width * h / bob_img.height)

    alice_img = alice_img.resize((alice_w, h))
    bob_img = bob_img.resize((bob_w, h))

    combined = Image.new("RGB", (alice_w + bob_w, h), "white")
    combined.paste(alice_img, (0, 0))
    combined.paste(bob_img, (alice_w, 0))

    combined.save(out_dir / f"video_{i:02d}_alice_bob.png", optimize=True)

print("Saved paired Alice/Bob plots in the 'paired' folder.")