  
    Gaussian kernel: G(x,y) = (1/(2πσ²)) * e^(-(x²+y²)/(2σ²))
    σ: standard deviation -> 5
    3 * 3: kernel size, should be odd and positive
    ex:
    1. calculate the kernal value at position (0, 0)
    x1(-1,-1): x²+y² = (-1)²+(-1)² = 2
    x2 ( 0,-1): x²+y² =  0²+(-1)² = 1
    x3 ( 1,-1): x²+y² =  1²+(-1)² = 2
    x4 (-1, 0): x²+y² = (-1)²+ 0²  = 1
    x5 ( 0, 0): x²+y² =  0²+ 0²  = 0
    x6 ( 1, 0): x²+y² =  1²+ 0²  = 1
    x7 (-1, 1): x²+y² = (-1)²+ 1²  = 2
    x8 ( 0, 1): x²+y² =  0²+ 1²  = 1
    x9 ( 1, 1): x²+y² =  1²+ 1²  = 2
    2. calculate the gaussian kernel value at each position
    1/(2πσ²) = 1/(2×3.1416×25) = 1/157.08 = 0.00637
    e^(-(x²+y²)/(2σ²)) = e^(-(x²+y²)/50)

    (-1, -1) (1/(2π*5²)) * e^(-2/50) = 00637 * e^(-0.04) = 0.00637 * 0.9608 = 0.00612    
    ( 0, -1) (1/(2π*5²)) * e^(-1/50) = 00637 * e^(-0.02) = 0.00637 * 0.9802 = 0.00624
    ( 1, -1) (1/(2π*5²)) * e^(-2/50) = 00637 * e^(-0.04) = 0.00637 * 0.9608 = 0.00612
    (-1, 0) (1/(2π*5²)) * e^(-1/50) = 00637 * e^(-0.02) = 0.00637 * 0.9802 = 0.00624
    ( 0, 0) (1/(2π*5²)) * e^(0) = 00637 * 1 = 0.00637
    ( 1, 0) (1/(2π*5²)) * e^(-1/50) = 00637 * e^(-0.02) = 0.00637 * 0.9802 = 0.00624
    (-1, 1) (1/(2π*5²)) * e^(-2/50) = 00637 * e^(-0.04) = 0.00637 * 0.9608 = 0.00612
    ( 0, 1) (1/(2π*5²)) * e^(-1/50) = 00637 * e^(-0.02) = 0.00637 * 0.9802 = 0.00624
    ( 1, 1) (1/(2π*5²)) * e^(-2/50) = 00637 * e^(-0.04) = 0.00637 * 0.9608 = 0.00612
    3. sum
    total = 4×0.00612 + 4×0.00624 + 0.00637
     = 0.02448 + 0.02496 + 0.00637
     = 0.05581
    4. normalize
    (-1,-1)	0.00612	0.00612/0.05581 = 0.1097
    (0,-1)	0.00624	0.00624/0.05581 = 0.1118
    (1,-1)	0.00612	0.00612/0.05581 = 0.1097
    (-1,0)	0.00624	0.00624/0.05581 = 0.1118
    (0,0)	0.00637	0.00637/0.05581 = 0.1142
    (1,0)	0.00624	0.00624/0.05581 = 0.1118
    (-1,1)	0.00612	0.00612/0.05581 = 0.1097
    (0,1)	0.00624	0.00624/0.05581 = 0.1118
    (1,1)	0.00612	0.00612/0.05581 = 0.1097
    5. final kernel matrix -> sigma=5, ksize=3
    0.1097  0.1118  0.1097
    0.1118  0.1142  0.1118
    0.1097  0.1118  0.1097
    
    vs sigma=3, ksize=3
    0.0621  0.0673  0.0621
    0.0673  0.0721  0.0673
    0.0621  0.0673  0.0621
    
    vs sigma=1, ksize=3
    0.0751  0.1238  0.0751
    0.1238  0.2042  0.1238
    0.0751  0.1238  0.0751
    
    the value of sigma controls the degree of smoothing
    bigger sigma -> more smoothing -> less noise but also less detail
    the matrix values in the kernel will be more spread out and less peaked
    
    sigma = 1 -> keep detail but more noise less smoothing
    sigma = 3 -> balance between detail and noise reduction
    sigma = 5 -> more smoothing, less noise but also less detail
    sigma = 10 -> very strong smoothing, significant noise reduction but also significant loss of detail
    