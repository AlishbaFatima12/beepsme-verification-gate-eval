# Calculus product rule differentiation

_Subject:_ **math** &middot; _Date:_ 2026-08-20 &middot; _Solved by Beepsme_

## Question 1

Differentiate x² · sin(x)

**Given:** f(x) = x² · sin(x)

**Diagram:**

```
f(x) = x² · sin(x)
         ↙           ↘
    u = x²        v = sin(x)
    u'= 2x         v'= cos(x)
         ↘           ↙
  f'(x) = u'v  +  uv'
        = 2x·sin(x) + x²·cos(x)
```

**Steps:**
1. Identify this as a product of two functions: u(x) = x² and v(x) = sin(x).
2. Recall the Product Rule: d/dx [u · v] = u' · v + u · v'
3. Differentiate u(x) = x²:  u'(x) = 2x
4. Differentiate v(x) = sin(x):  v'(x) = cos(x)
5. Apply the product rule: f'(x) = u'·v + u·v' = (2x)·sin(x) + x²·cos(x)
6. Simplify (already in simplest form): f'(x) = 2x·sin(x) + x²·cos(x)

**Answer:** f'(x) = 2x·sin(x) + x²·cos(x)

_Why this works:_ The Product Rule must be used here because both x² and sin(x) are functions of x — a common mistake is to just multiply the individual derivatives (2x · cos(x)), which is incorrect.
