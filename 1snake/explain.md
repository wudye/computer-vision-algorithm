active_contour(image, snake, alpha=0.01, beta=0.1, w_line=0, w_edge=1, gamma=0.01,
        bc='periodic', max_px_move=1.0, max_iterations=2500, convergence=0.1)

    Parameters
    ----------
    image : (N, M) or (N, M, 3) ndarray
        Input image.
    snake : (N, 2) ndarray
        Initial snake coordinates. For periodic boundary conditions, endpoints
        must not be duplicated.
    alpha : float, optional
        Snake length shape parameter. Higher values makes snake contract
        faster.
    beta : float, optional
        Snake smoothness shape parameter. Higher values makes snake smoother.
    w_line : float, optional
        Controls attraction to brightness. Use negative values to attract toward
        dark regions.
    w_edge : float, optional
        Controls attraction to edges. Use negative values to repel snake from
        edges.
    gamma : float, optional
        Explicit time stepping parameter.
    bc : {'periodic', 'free', 'fixed'}, optional
        Boundary conditions for worm. 'periodic' attaches the two ends of the
        snake, 'fixed' holds the end-points in place, and 'free' allows free
        movement of the ends. 'fixed' and 'free' can be combined by parsing
        'fixed-free', 'free-fixed'. Parsing 'fixed-fixed' or 'free-free'
        yields same behaviour as 'fixed' and 'free', respectively.
    max_px_move : float, optional
        Maximum pixel distance to move per iteration.
    max_iterations : int, optional
        Maximum iterations to optimize snake shape.
    convergence: float, optional
        Convergence criteria.

    Returns
    -------
    snake : (N, 2) ndarray
        Optimised snake, same shape as input parameter.

    References
    ----------
    .. [1]  Kass, M.; Witkin, A.; Terzopoulos, D. "Snakes: Active contour
            models". International Journal of Computer Vision 1 (4): 321
            (1988). DOI:`10.1007/BF00133570`
