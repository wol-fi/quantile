# Explicit quantile representations

A distribution function may be explicit while its inverse has no familiar closed form. This project gives exact integral representations of quantile functions. They use forward functions only, along a contour chosen without knowing the quantile. One contour serves every probability level.

**Paper:** Wolfgang Schadner (2026), *A general technique for explicit quantile representations*, [working paper](paper/working_paper.pdf).

## Main formula

For a working distribution with density $g=\mathcal P'>0$ on an open interval $I$, CDF $\mathcal P$, and quantile $\mathcal Q=\mathcal P^{-1}$,

$$
\mathcal Q(p)=\frac{1}{\pi}\operatorname{Im}\int_I\kappa'(x)\operatorname{Log}\left(\frac{\mathcal P(\kappa(x))-p}{1-p}\right)\mathrm{d}x,
\qquad \kappa(x)=x+\mathrm{i}h(x),\qquad 0<p<1.
$$

The height $h>0$ is built from the density and does not depend on $p$. The idea is to keep the real part of the density positive on each vertical segment from $x$ to $x+\mathrm{i}h(x)$. Then

$$
\operatorname{Im}\mathcal P(x+\mathrm{i}y)=\int_0^y\operatorname{Re}g(x+\mathrm{i}t)\mathrm{d}t>0,\qquad 0<y\le h(x),
$$

so the complex CDF cannot equal a real probability above the axis, and the contour isolates the real quantile. Endpoint estimates complete the argument. The density form of the same identity is

$$
\mathcal Q(p)=-\frac{1}{\pi}\operatorname{Im}\int_I\kappa(x)\kappa'(x)\frac{g(\kappa(x))}{\mathcal P(\kappa(x))-p}\mathrm{d}x.
$$

Every strictly positive real-analytic density on an open interval admits such a contour after a known change of coordinate to $(0,1)$, with no moment assumption (Theorem 2.1). The logarithmic form needs a vanishing boundary term in addition, which is checked separately. Appendix B has the proofs and endpoint conditions.

## Explicit contours

Table 1 of the paper lists the working density $g$, the height $h$, and the map back to the original quantile $\widetilde{\mathcal Q}$. Here $\mathcal Q$ is the working quantile, $\mu$ and $\sigma>0$ are location and scale, and shape symbols are local to each row.

| Distribution | $g(x)$ | $h(x)$ | $\widetilde{\mathcal Q}(p)$ |
| --- | --- | --- | --- |
| **Generalized beta** | $\dfrac{x^{a-1}(1-x)^{b-1}}{\mathrm B(a,b)}$ | $\dfrac{x(1-x)}{1+\lvert a-1\rvert(1-x)+\lvert b-1\rvert x}$ | $\sigma\left[\dfrac{\mathcal Q(p)}{1-\rho\mathcal Q(p)}\right]^{1/c}$ |
| **Generalized gamma** | $\dfrac{x^{a-1}e^{-x}}{\Gamma(a)}$ | $\dfrac{x}{1+\lvert a-1\rvert+x}$ | $\sigma[\mathcal Q(p)]^{1/c}$ |
| **Generalized inverse Gaussian** | $\dfrac{x^{a-1}}{2K_a(b)}\exp\left[-\dfrac{b}{2}\left(x+\dfrac{1}{x}\right)\right]$ | $\dfrac{x^2}{b+2\lvert a-1\rvert x+bx^2}$ | $\sigma\mathcal Q(p)$ |
| **Generalized hyperbolic** | $g_{\mathrm{GH}}(x)$ | $h_{\mathrm{GH}}$ | $\mu+\sigma\mathcal Q(p)$ |
| **Generalized extreme value**, $a\ne0$ | $e^{-x}$ | $\dfrac{x}{1+x}$ | $\mu+\dfrac{\sigma}{a}\left([\mathcal Q(1-p)]^{-a}-1\right)$ |
| **Generalized extreme value**, $a=0$ | $e^{-x}$ | $\dfrac{x}{1+x}$ | $\mu-\sigma\log\mathcal Q(1-p)$ |
| Normal | $\phi(x)$ | $\dfrac{1}{2(1+x^2)}$ | $\mu+\sigma\mathcal Q(p)$ |
| Student's t | $t_a(x)$ | $\dfrac{\sqrt a}{2(a+1)}$ | $\mu+\sigma\mathcal Q(p)$ |
| Cauchy | $\dfrac{1}{\pi(1+x^2)}$ | $1/4$ | $\mu+\sigma\mathcal Q(p)$ |
| Variance gamma | $g_\pm(x)$ | $h_{\mathrm{VG}}(x)$ | $\mu+\sigma\mathcal Q(p)$ |
| Log-normal | $\phi(x)$ | $\dfrac{1}{2(1+x^2)}$ | $\sigma\exp[a\mathcal Q(p)]$ |

Notes:

- $\phi$ is the standard normal density, and $t_a$ is the Student density with $a>0$ degrees of freedom. Student and Cauchy use the bounded observable $G(z)=\arctan z$ in place of $\kappa$ in the density form (Eq. A.21), which covers laws without a first moment.
- For generalized hyperbolic, $g_{\mathrm{GH}}$ is the standardized density with $a\in\mathbb R$ and $b>\lvert d\rvert$, and, with $D=3/2+\lvert a-1/2\rvert-a$,

$$
h_{\mathrm{GH}}=\min\left(\frac12,\frac{e^{-b(1-\sqrt3/2)}(\sqrt3/2)^{D}}{2\lvert d\rvert+4b/\sqrt3+8D/3}\right).
$$

- For variance gamma, $g_\pm$ are the two half-axis subdensities on $(0,\infty)$, with $a>-1/2$ and $d\in\mathbb R$. The sign of $p-\mathcal P(0)$ selects the half-axis, and $\lvert p-\mathcal P(0)\rvert$ sets the target mass (Theorem A.2). The height is

$$
h_{\mathrm{VG}}(x)=\frac{x}{8\cdot2^{\lvert a\rvert}\left[1+2\lvert a\rvert+\left(\sqrt{1+d^2}+\lvert d\rvert\right)x\right]}.
$$

Appendix A has the derivations.

## Black-Scholes implied volatility

Let $\mathsf c=\mathsf C/(\mathsf D\mathsf F)\in(0,1)$ be the call price normalized by the discount factor and forward, with observed value $\bar{\mathsf c}$. Let $\mathsf k=\log(\mathsf K/\mathsf F)>0$ be the log-moneyness, $\mathsf T>0$ the maturity, and $v=\sigma_{\mathrm{BS}}\sqrt{\mathsf T}$ the total volatility. With $\Phi$ the standard normal CDF, the pricing formula is

$$
\mathsf c(\mathsf k,v)=\Phi\left(-\frac{\mathsf k}{v}+\frac{v}{2}\right)-e^{\mathsf k}\Phi\left(-\frac{\mathsf k}{v}-\frac{v}{2}\right).
$$

Implied volatility then has the exact finite-interval representation

$$
\sigma_{\mathrm{BS}}=\frac{1}{\pi\sqrt{\mathsf T}}\operatorname{Im}\int_0^1\nu_{\mathsf k}'(t)\operatorname{Log}\left(\frac{\mathsf c(\mathsf k,\nu_{\mathsf k}(t))-\bar{\mathsf c}}{1-\bar{\mathsf c}}\right)\mathrm{d}t,
$$

with principal branches and the contour

$$
\nu_{\mathsf k}(t)=\sqrt{\kappa_{\mathsf k}\left(\frac{t}{1-t}\right)},\qquad
\kappa_{\mathsf k}(x)=x+\mathrm{i}\frac{x^2}{\mathsf k^2+x+x^2/4}.
$$

It involves only elementary functions and the error function, and it uses no root finding. The contour does not depend on $\bar{\mathsf c}$ or $\mathsf T$. Put-call parity, $\mathsf c(\mathsf k,v)=1-e^{\mathsf k}+e^{\mathsf k}\mathsf c(-\mathsf k,v)$, reduces $\mathsf k<0$ to the positive branch, and $\mathsf k=0$ reduces to a normal-CDF expression.

Appendix E gives the corresponding representation of Bachelier (normal) implied volatility.

## Repository

```text
paper/
  working_paper.pdf         the paper with its online appendix
numerical_tests/
  README.md                 setup, results, and how to rerun
  distribution_quantiles/   quantile integrals for twelve working laws (Appendix D.1)
  implied_volatility/       the Black-Scholes integral (Appendix D.2)
requirements.txt            Python packages used by the tests
```

The tests compare direct contour-integral evaluations with independently computed reference values. All 96 relative errors in the distribution test are below $10^{-17}$, and the largest relative volatility error in the 42-case Black-Scholes grid is $2.22\times10^{-15}$. These figures measure agreement on the test sets; they are not uniform error bounds. See [numerical_tests](numerical_tests) for details.

## Citation

```bibtex
@unpublished{Schadner2026Technique,
  author = {Schadner, Wolfgang},
  title  = {A general technique for explicit quantile representations},
  note   = {Working paper},
  year   = {2026},
  url    = {https://github.com/wol-fi/quantile}
}
```

## License

The code is released under the MIT license, see [LICENSE](LICENSE).
