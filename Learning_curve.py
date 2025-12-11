import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re

# We manually add Epoch 1 based on your memory
# And then paste the Epoch 9-100 logs below it
log_data = """
Epoch 1 Summary:
Train Loss: 22.0000 (pos: 12.0000, neg: 10.0000)
Val Loss:   21.5000 (pos: 11.5000, neg: 10.0000)

Epoch 9 Summary:
Train Loss: 11.4680 (pos: 6.1838, neg: 5.2842)
Val Loss:   12.3082 (pos: 7.1471, neg: 5.1611)

Epoch 10 Summary:
Train Loss: 11.4092 (pos: 6.0386, neg: 5.3707)
Val Loss:   11.7604 (pos: 6.3746, neg: 5.3858)

Epoch 11 Summary:
Train Loss: 11.0312 (pos: 5.6746, neg: 5.3566)
Val Loss:   12.0633 (pos: 6.5185, neg: 5.5448)

Epoch 12 Summary:
Train Loss: 11.0798 (pos: 5.8302, neg: 5.2496)
Val Loss:   11.5558 (pos: 6.6252, neg: 4.9306)

Epoch 13 Summary:
Train Loss: 10.9223 (pos: 5.8224, neg: 5.0999)
Val Loss:   11.4172 (pos: 5.9474, neg: 5.4698)

Epoch 14 Summary:
Train Loss: 10.8912 (pos: 5.7316, neg: 5.1596)
Val Loss:   11.3381 (pos: 6.2919, neg: 5.0462)

Epoch 15 Summary:
Train Loss: 10.4273 (pos: 5.5895, neg: 4.8378)
Val Loss:   10.8201 (pos: 5.8603, neg: 4.9597)

Epoch 16 Summary:
Train Loss: 10.5476 (pos: 5.5499, neg: 4.9978)
Val Loss:   10.4229 (pos: 5.6201, neg: 4.8028)

Epoch 17 Summary:
Train Loss: 10.3141 (pos: 5.3461, neg: 4.9679)
Val Loss:   10.5111 (pos: 5.6297, neg: 4.8814)

Epoch 18 Summary:
Train Loss: 10.3209 (pos: 5.5405, neg: 4.7803)
Val Loss:   10.4498 (pos: 5.9636, neg: 4.4862)

Epoch 19 Summary:
Train Loss: 10.1847 (pos: 5.3655, neg: 4.8192)
Val Loss:   10.2691 (pos: 5.9880, neg: 4.2811)

Epoch 20 Summary:
Train Loss: 10.0899 (pos: 5.4585, neg: 4.6314)
Val Loss:   10.1976 (pos: 5.4505, neg: 4.7471)

Epoch 21 Summary:
Train Loss: 10.0668 (pos: 5.4733, neg: 4.5934)
Val Loss:   10.0618 (pos: 5.5769, neg: 4.4849)

Epoch 22 Summary:
Train Loss: 10.2168 (pos: 5.4948, neg: 4.7220)
Val Loss:   10.7111 (pos: 5.8522, neg: 4.8589)

Epoch 23 Summary:
Train Loss: 10.0470 (pos: 5.5211, neg: 4.5259)
Val Loss:   9.9095 (pos: 5.5663, neg: 4.3432)

Epoch 24 Summary:
Train Loss: 9.9732 (pos: 5.5715, neg: 4.4017)
Val Loss:   10.2805 (pos: 5.8403, neg: 4.4401)

Epoch 25 Summary:
Train Loss: 9.6885 (pos: 5.4133, neg: 4.2752)
Val Loss:   10.0538 (pos: 5.5752, neg: 4.4786)

Epoch 26 Summary:
Train Loss: 9.7527 (pos: 5.3332, neg: 4.4196)
Val Loss:   10.4437 (pos: 5.7373, neg: 4.7064)

Epoch 27 Summary:
Train Loss: 9.7361 (pos: 5.2553, neg: 4.4807)
Val Loss:   10.5329 (pos: 5.8977, neg: 4.6352)

Epoch 28 Summary:
Train Loss: 9.6607 (pos: 5.3883, neg: 4.2724)
Val Loss:   9.8470 (pos: 5.6356, neg: 4.2115)

Epoch 29 Summary:
Train Loss: 9.7149 (pos: 5.2091, neg: 4.5058)
Val Loss:   10.0437 (pos: 5.5052, neg: 4.5385)

Epoch 30 Summary:
Train Loss: 9.6792 (pos: 5.2463, neg: 4.4329)
Val Loss:   10.3293 (pos: 5.6562, neg: 4.6731)

Epoch 31 Summary:
Train Loss: 9.4370 (pos: 5.3461, neg: 4.0909)
Val Loss:   9.6754 (pos: 5.5987, neg: 4.0767)

Epoch 32 Summary:
Train Loss: 9.4573 (pos: 5.2081, neg: 4.2492)
Val Loss:   9.8719 (pos: 5.4588, neg: 4.4131)

Epoch 33 Summary:
Train Loss: 9.6960 (pos: 5.3392, neg: 4.3568)
Val Loss:   9.5723 (pos: 5.7036, neg: 3.8686)

Epoch 34 Summary:
Train Loss: 9.6675 (pos: 5.4179, neg: 4.2495)
Val Loss:   10.4884 (pos: 5.9856, neg: 4.5028)

Epoch 35 Summary:
Train Loss: 9.5191 (pos: 5.3082, neg: 4.2109)
Val Loss:   9.8553 (pos: 5.6016, neg: 4.2537)

Epoch 36 Summary:
Train Loss: 9.5133 (pos: 5.2764, neg: 4.2369)
Val Loss:   10.2407 (pos: 5.2668, neg: 4.9739)

Epoch 37 Summary:
Train Loss: 9.8170 (pos: 5.3279, neg: 4.4892)
Val Loss:   9.6862 (pos: 5.5297, neg: 4.1565)

Epoch 38 Summary:
Train Loss: 9.4186 (pos: 5.2090, neg: 4.2096)
Val Loss:   9.8805 (pos: 5.1273, neg: 4.7532)

Epoch 39 Summary:
Train Loss: 9.5628 (pos: 5.2882, neg: 4.2746)
Val Loss:   9.9305 (pos: 5.6910, neg: 4.2395)

Epoch 40 Summary:
Train Loss: 9.1610 (pos: 5.2266, neg: 3.9344)
Val Loss:   9.7476 (pos: 5.4677, neg: 4.2798)

Epoch 41 Summary:
Train Loss: 8.9746 (pos: 5.1203, neg: 3.8544)
Val Loss:   9.4091 (pos: 5.4614, neg: 3.9476)

Epoch 42 Summary:
Train Loss: 8.7760 (pos: 5.0724, neg: 3.7036)
Val Loss:   9.1139 (pos: 5.4366, neg: 3.6773)

Epoch 43 Summary:
Train Loss: 8.8240 (pos: 5.0711, neg: 3.7529)
Val Loss:   9.2662 (pos: 5.1706, neg: 4.0956)

Epoch 44 Summary:
Train Loss: 8.7200 (pos: 4.9415, neg: 3.7785)
Val Loss:   8.9749 (pos: 5.4211, neg: 3.5538)

Epoch 45 Summary:
Train Loss: 8.8382 (pos: 5.0273, neg: 3.8109)
Val Loss:   8.7575 (pos: 4.9316, neg: 3.8259)

Epoch 46 Summary:
Train Loss: 8.7780 (pos: 5.0533, neg: 3.7247)
Val Loss:   8.7182 (pos: 5.0173, neg: 3.7010)

Epoch 47 Summary:
Train Loss: 8.7212 (pos: 4.9413, neg: 3.7799)
Val Loss:   8.8265 (pos: 5.1138, neg: 3.7127)

Epoch 48 Summary:
Train Loss: 8.8939 (pos: 4.9776, neg: 3.9163)
Val Loss:   9.1501 (pos: 4.9849, neg: 4.1652)

Epoch 49 Summary:
Train Loss: 8.8031 (pos: 4.9552, neg: 3.8479)
Val Loss:   9.0008 (pos: 5.0986, neg: 3.9023)

Epoch 50 Summary:
Train Loss: 8.6915 (pos: 4.8468, neg: 3.8448)
Val Loss:   8.8662 (pos: 4.8014, neg: 4.0648)

Epoch 51 Summary:
Train Loss: 8.7705 (pos: 4.9416, neg: 3.8289)
Val Loss:   8.9041 (pos: 5.0774, neg: 3.8267)

Epoch 52 Summary:
Train Loss: 8.6923 (pos: 4.9373, neg: 3.7550)
Val Loss:   8.5673 (pos: 4.9982, neg: 3.5691)

Epoch 53 Summary:
Train Loss: 8.5879 (pos: 4.8249, neg: 3.7630)
Val Loss:   8.7955 (pos: 4.8083, neg: 3.9871)

Epoch 54 Summary:
Train Loss: 8.8389 (pos: 4.9700, neg: 3.8690)
Val Loss:   9.6778 (pos: 5.2890, neg: 4.3888)

Epoch 55 Summary:
Train Loss: 8.7875 (pos: 4.9609, neg: 3.8266)
Val Loss:   8.9919 (pos: 5.3466, neg: 3.6453)

Epoch 56 Summary:
Train Loss: 8.6672 (pos: 4.9565, neg: 3.7106)
Val Loss:   9.1618 (pos: 5.1709, neg: 3.9910)

Epoch 57 Summary:
Train Loss: 8.8758 (pos: 5.0672, neg: 3.8086)
Val Loss:   9.1778 (pos: 5.3057, neg: 3.8721)

Epoch 58 Summary:
Train Loss: 8.7447 (pos: 4.9746, neg: 3.7701)
Val Loss:   8.5345 (pos: 4.9482, neg: 3.5863)

Epoch 59 Summary:
Train Loss: 8.5990 (pos: 4.8902, neg: 3.7087)
Val Loss:   9.0765 (pos: 5.0960, neg: 3.9805)

Epoch 60 Summary:
Train Loss: 8.6231 (pos: 4.9053, neg: 3.7178)
Val Loss:   8.6925 (pos: 5.1171, neg: 3.5754)

Epoch 61 Summary:
Train Loss: 8.6672 (pos: 4.9382, neg: 3.7290)
Val Loss:   8.7592 (pos: 5.3115, neg: 3.4478)

Epoch 62 Summary:
Train Loss: 8.6639 (pos: 4.9358, neg: 3.7282)
Val Loss:   8.6382 (pos: 4.9080, neg: 3.7302)

Epoch 63 Summary:
Train Loss: 8.5334 (pos: 4.9511, neg: 3.5822)
Val Loss:   8.6203 (pos: 5.1251, neg: 3.4952)

Epoch 64 Summary:
Train Loss: 8.7785 (pos: 5.0945, neg: 3.6840)
Val Loss:   9.0475 (pos: 5.4169, neg: 3.6306)

Epoch 65 Summary:
Train Loss: 8.5446 (pos: 4.8989, neg: 3.6457)
Val Loss:   8.8646 (pos: 4.8369, neg: 4.0277)

Epoch 66 Summary:
Train Loss: 8.4699 (pos: 4.8427, neg: 3.6272)
Val Loss:   8.6408 (pos: 5.1310, neg: 3.5099)

Epoch 67 Summary:
Train Loss: 8.4708 (pos: 4.7863, neg: 3.6845)
Val Loss:   8.4256 (pos: 4.9497, neg: 3.4758)

Epoch 68 Summary:
Train Loss: 8.3257 (pos: 4.7988, neg: 3.5269)
Val Loss:   8.7089 (pos: 4.9358, neg: 3.7730)

Epoch 69 Summary:
Train Loss: 8.3568 (pos: 4.8307, neg: 3.5261)
Val Loss:   8.5846 (pos: 4.9209, neg: 3.6636)

Epoch 70 Summary:
Train Loss: 8.2279 (pos: 4.7895, neg: 3.4384)
Val Loss:   8.6834 (pos: 4.8693, neg: 3.8142)

Epoch 71 Summary:
Train Loss: 8.2500 (pos: 4.8226, neg: 3.4274)
Val Loss:   8.5454 (pos: 5.0472, neg: 3.4982)

Epoch 72 Summary:
Train Loss: 8.3083 (pos: 4.8452, neg: 3.4631)
Val Loss:   8.4463 (pos: 4.9819, neg: 3.4644)

Epoch 73 Summary:
Train Loss: 8.1627 (pos: 4.8036, neg: 3.3592)
Val Loss:   8.2447 (pos: 4.8248, neg: 3.4198)

Epoch 74 Summary:
Train Loss: 8.2797 (pos: 4.7210, neg: 3.5587)
Val Loss:   8.5848 (pos: 4.8860, neg: 3.6988)

Epoch 75 Summary:
Train Loss: 8.3639 (pos: 4.7820, neg: 3.5818)
Val Loss:   8.3731 (pos: 4.7634, neg: 3.6097)

Epoch 76 Summary:
Train Loss: 8.1783 (pos: 4.7390, neg: 3.4392)
Val Loss:   8.3418 (pos: 4.7702, neg: 3.5716)

Epoch 77 Summary:
Train Loss: 8.2186 (pos: 4.7602, neg: 3.4584)
Val Loss:   8.3579 (pos: 4.7139, neg: 3.6441)

Epoch 78 Summary:
Train Loss: 8.2394 (pos: 4.7827, neg: 3.4567)
Val Loss:   8.6660 (pos: 4.8751, neg: 3.7910)

Epoch 79 Summary:
Train Loss: 8.2904 (pos: 4.8386, neg: 3.4518)
Val Loss:   8.7445 (pos: 4.8078, neg: 3.9367)

Epoch 80 Summary:
Train Loss: 8.1475 (pos: 4.6768, neg: 3.4708)
Val Loss:   8.3401 (pos: 4.7177, neg: 3.6224)

Epoch 81 Summary:
Train Loss: 8.1817 (pos: 4.7030, neg: 3.4787)
Val Loss:   8.1153 (pos: 4.7319, neg: 3.3835)

Epoch 82 Summary:
Train Loss: 8.1027 (pos: 4.6677, neg: 3.4350)
Val Loss:   8.0921 (pos: 4.6514, neg: 3.4407)

Epoch 83 Summary:
Train Loss: 8.1769 (pos: 4.7085, neg: 3.4684)
Val Loss:   8.3399 (pos: 4.8932, neg: 3.4467)

Epoch 84 Summary:
Train Loss: 8.1864 (pos: 4.7918, neg: 3.3946)
Val Loss:   8.4806 (pos: 5.0687, neg: 3.4119)

Epoch 85 Summary:
Train Loss: 8.1266 (pos: 4.7227, neg: 3.4039)
Val Loss:   7.9073 (pos: 4.6050, neg: 3.3023)

Epoch 86 Summary:
Train Loss: 8.0534 (pos: 4.6525, neg: 3.4009)
Val Loss:   8.7284 (pos: 4.8797, neg: 3.8487)

Epoch 87 Summary:
Train Loss: 8.0589 (pos: 4.6943, neg: 3.3646)
Val Loss:   7.7891 (pos: 4.6196, neg: 3.1695)

Epoch 88 Summary:
Train Loss: 8.0344 (pos: 4.7228, neg: 3.3116)
Val Loss:   8.2821 (pos: 4.7708, neg: 3.5112)

Epoch 89 Summary:
Train Loss: 8.1826 (pos: 4.6439, neg: 3.5386)
Val Loss:   8.5820 (pos: 4.8307, neg: 3.7513)

Epoch 90 Summary:
Train Loss: 7.9899 (pos: 4.5844, neg: 3.4055)
Val Loss:   8.3549 (pos: 4.7319, neg: 3.6231)

Epoch 91 Summary:
Train Loss: 8.0825 (pos: 4.6031, neg: 3.4795)
Val Loss:   8.4077 (pos: 4.8902, neg: 3.5176)

Epoch 92 Summary:
Train Loss: 8.0108 (pos: 4.6923, neg: 3.3185)
Val Loss:   8.0277 (pos: 4.7289, neg: 3.2989)

Epoch 93 Summary:
Train Loss: 8.1434 (pos: 4.6621, neg: 3.4812)
Val Loss:   8.3758 (pos: 4.6519, neg: 3.7239)

Epoch 94 Summary:
Train Loss: 7.9438 (pos: 4.5964, neg: 3.3474)
Val Loss:   8.6439 (pos: 4.7866, neg: 3.8572)

Epoch 95 Summary:
Train Loss: 8.0317 (pos: 4.6381, neg: 3.3936)
Val Loss:   8.0436 (pos: 4.8210, neg: 3.2226)

Epoch 96 Summary:
Train Loss: 8.0417 (pos: 4.6684, neg: 3.3733)
Val Loss:   7.8093 (pos: 4.5131, neg: 3.2962)

Epoch 97 Summary:
Train Loss: 8.0444 (pos: 4.6833, neg: 3.3611)
Val Loss:   8.2522 (pos: 4.8055, neg: 3.4468)

Epoch 98 Summary:
Train Loss: 7.9586 (pos: 4.6305, neg: 3.3281)
Val Loss:   7.6515 (pos: 4.6108, neg: 3.0406)

Epoch 99 Summary:
Train Loss: 7.9557 (pos: 4.6501, neg: 3.3056)
Val Loss:   7.6797 (pos: 4.6461, neg: 3.0336)

Epoch 100 Summary:
Train Loss: 7.9990 (pos: 4.6087, neg: 3.3902)
Val Loss:   8.2377 (pos: 4.5632, neg: 3.6745)
"""

# Parsing logic
data = []
pattern = r"Epoch (\d+) Summary.*?Train Loss:\s+([\d\.]+)\s+\(pos:\s+([\d\.]+),\s+neg:\s+([\d\.]+)\).*?Val Loss:\s+([\d\.]+)\s+\(pos:\s+([\d\.]+),\s+neg:\s+([\d\.]+)\)"

chunks = log_data.split("Epoch ")
for chunk in chunks:
    if "Summary" in chunk:
        full_chunk = "Epoch " + chunk
        match = re.search(pattern, full_chunk, re.DOTALL)
        if match:
            data.append({
                'Epoch': int(match.group(1)),
                'Train_Loss': float(match.group(2)),
                'Train_Pos': float(match.group(3)),
                'Train_Neg': float(match.group(4)),
                'Val_Loss': float(match.group(5)),
                'Val_Pos': float(match.group(6)),
                'Val_Neg': float(match.group(7))
            })

df = pd.DataFrame(data)

# Plotting
sns.set_style("whitegrid")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Total Loss
ax1.plot(df['Epoch'], df['Train_Loss'], label='Training Loss', color='blue', alpha=0.6, linewidth=1.5)
ax1.plot(df['Epoch'], df['Val_Loss'], label='Validation Loss', color='orange', linewidth=2)
# Smoothing for clearer trend
z = pd.Series(df['Val_Loss']).rolling(window=10, min_periods=1).mean()
ax1.plot(df['Epoch'], z, label='Val Loss Trend', color='red', linestyle='--', linewidth=2)

ax1.set_title('Model Convergence (Full 100 Epochs)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Contrastive Loss')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.7)

# Plot 2: Embeddings Separation
ax2.plot(df['Epoch'], df['Val_Pos'], label='Positive Distance (Same Song)', color='green')
ax2.plot(df['Epoch'], df['Val_Neg'], label='Negative Distance (Diff Song/Time)', color='red')
ax2.axhline(y=24, color='gray', linestyle=':', label='Target Margin (24 bits)')

ax2.set_title('Fingerprint Separation Quality', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Hamming Distance (Bits)')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('final_presentation_curves.png', dpi=300)
plt.show()