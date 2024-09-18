import numpy as np

def calculate_hsv_manual(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx - mn
    
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g - b) / df) + 360) % 360
    elif mx == g:
        h = (60 * ((b - r) / df) + 120) % 360
    elif mx == b:
        h = (60 * ((r - g) / df) + 240) % 360
    
    s = 0 if mx == 0 else (df / mx)
    v = mx
    
    return h, s, v

# Menggunakan nilai rata-rata yang sama dengan perhitungan manual
r_avg = 99.33
g_avg = 79.0
b_avg = 72.56

# Menghitung HSV
h_avg, s_avg, v_avg = calculate_hsv_manual(r_avg, g_avg, b_avg)
print(f"H: {h_avg}, S: {s_avg}, V: {v_avg}")
