blues = [
["dark navy blue", "#00022e"],
["cobalt blue", "#030aa7" ],
]

reds = [
[ "brik red", "#8f1402"],
[  "verbeillon","#f4320c"],
[  "blood","#770001"]
]

others = [
["chocolate","#3d1c02"],
["dark coral","#cf5242" ],
["cement","#a5a391"],
["aqua green","#12e193" ],
["petrol","#005f6a" ],
]

yellows = [
["amber","#feb408" ],
["golden","#f5bf03" ],
["golden rod","#fac205" ],
[  "sunny yellow","#fff917"]
]

greens = [
["dark olive green","#3c4d03"  ],
["avocado","#87a922" ],
["irish green","#019529"]
]

# Create a dictinary !!

all_colors = []
all_colors.extend(blues)
all_colors.extend(greens)
all_colors.extend(yellows)
all_colors.extend(reds)
all_colors.extend(others)

cd = dict(all_colors)  # Color dictionary