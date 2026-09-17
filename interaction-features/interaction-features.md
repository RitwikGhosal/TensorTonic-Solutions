## What Are Interaction Features?

Interaction features are new features created by combining two or more existing features, capturing relationships that individual features cannot represent alone. They allow models to learn how the effect of one feature depends on the value of another.

The most common form is the product of two features, but interactions can also involve ratios, sums, differences, or more complex combinations.

---

## Why Create Interaction Features?

**1. Capture non-additive effects:**

Sometimes the combined effect of two features is not the sum of their individual effects.

**Example:** Price alone and Quality alone might each have weak effects, but Price * Quality (value for money) might be highly predictive.

**2. Help linear models:**

Linear models assume features affect the target independently. Interaction features allow them to model dependencies.

**3. Encode domain knowledge:**

Known relationships between features can be explicitly modeled.

---

## Mathematical Motivation

Consider a linear model:

$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2
$$

This assumes $x_1$ and $x_2$ affect $y$ independently.

With an interaction term:

$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 (x_1 \cdot x_2)
$$

Now the effect of $x_1$ on $y$ depends on the value of $x_2$:

$$
\frac{\partial y}{\partial x_1} = \beta_1 + \beta_3 x_2
$$

---

## Types of Interaction Features

**1. Multiplicative (product):**

$$
x_{new} = x_1 \cdot x_2
$$

Most common type, captures how features scale together.

**2. Division (ratio):**

$$
x_{new} = \frac{x_1}{x_2}
$$

Useful for rate-based features.

**3. Addition:**

$$
x_{new} = x_1 + x_2
$$

Useful when combined magnitude matters.

**4. Difference:**

$$
x_{new} = x_1 - x_2
$$

Captures relative position or change.

---

## Multiplicative Interaction Example

**Features:**

- $x_1$: Square footage of house
- $x_2$: Price per square foot in the neighborhood

**Interaction:**

$$
x_{new} = x_1 \cdot x_2 = \text{estimated house value}
$$

**Sample calculation:**

- House: 2000 sq ft
- Neighborhood: $150/sq ft
- Interaction: 2000 * 150 = $300,000

This interaction directly gives the house value, which neither feature alone provides.

---

## Ratio Interaction Example

**Features:**

- $x_1$: Debt amount
- $x_2$: Income

**Interaction:**

$$
x_{new} = \frac{x_1}{x_2} = \text{debt-to-income ratio}
$$

**Sample calculation:**

- Debt: $50,000
- Income: $100,000
- Ratio: 0.5

This ratio is a key indicator in credit scoring, more predictive than either value alone.

---

## Difference Interaction Example

**Features:**

- $x_1$: Current temperature
- $x_2$: Historical average temperature

**Interaction:**

$$
x_{new} = x_1 - x_2 = \text{temperature anomaly}
$$

**Sample calculation:**

- Current: 85°F
- Historical average: 75°F
- Anomaly: +10°F

This captures whether current conditions are unusual.

---

## Categorical Interaction Features

For categorical features, interaction means combining categories:

**Features:**

- Color: {Red, Blue, Green}
- Size: {Small, Medium, Large}

**Interaction:** Color_Size

- Red_Small, Red_Medium, Red_Large
- Blue_Small, Blue_Medium, Blue_Large
- Green_Small, Green_Medium, Green_Large

This creates 9 combined categories from 3 + 3 original categories.

---

## Mixed Interactions: Categorical and Numerical

**Features:**

- $x_1$: Region (categorical: North, South, East, West)
- $x_2$: Temperature (numerical)

**Interaction:** Create separate temperature features for each region:

- Temperature_North
- Temperature_South
- Temperature_East
- Temperature_West

This allows the temperature effect to vary by region.

---

## Pairwise Interactions

Given $n$ features, create all pairwise products:

**Features:** $x_1, x_2, x_3$

**Pairwise interactions:**

- $x_1 \cdot x_2$
- $x_1 \cdot x_3$
- $x_2 \cdot x_3$

**Number of pairwise interactions:**

$$
\binom{n}{2} = \frac{n(n-1)}{2}
$$

For 10 features: $\frac{10 \cdot 9}{2} = 45$ interaction features.

---

## Higher-Order Interactions

Interactions can involve more than two features:

**Three-way interaction:**

$$
x_{new} = x_1 \cdot x_2 \cdot x_3
$$

**Example:** The effect of price on sales depends on both advertising spend and season.

**Caution:** Higher-order interactions grow combinatorially and can lead to overfitting.

Number of k-way interactions from n features:

$$
\binom{n}{k} = \frac{n!}{k!(n-k)!}
$$

---

## Feature Explosion Problem

Creating all possible interactions can lead to too many features:

**10 features with all pairwise:** 45 new features

**10 features with all up to 3-way:**

$$
\binom{10}{2} + \binom{10}{3} = 45 + 120 = 165 \text{ new features}
$$

**Mitigation strategies:**

1. Use domain knowledge to select important interactions
2. Apply feature selection after creating interactions
3. Use regularization (Lasso) to remove unhelpful interactions
4. Use tree-based models that learn interactions automatically

---

## Domain-Driven Interactions

Use knowledge of the problem to create meaningful interactions:

**E-commerce:**

- Items_in_cart * Avg_item_price = Cart_value
- Visits * Conversion_rate = Expected_purchases

**Healthcare:**

- Weight / Height^2 = BMI
- Systolic - Diastolic = Pulse_pressure

**Finance:**

- Principal * Rate * Time = Interest
- Revenue - Expenses = Profit

---

## Worked Example: Complete Process

**Original features:**

- Age: 35
- Income: $80,000
- Education_years: 16

**Interactions created:**

1. Age * Income = 35 * 80000 = 2,800,000
2. Age * Education = 35 * 16 = 560
3. Income * Education = 80000 * 16 = 1,280,000
4. Income / Age = 80000 / 35 = 2,286 (income per year of age)
5. Income / Education = 80000 / 16 = 5,000 (income per year of education)

**Result:** 5 original features become 8 features (3 original + 5 interactions)

---

## Scaling Considerations

Interaction features can have very different scales:

**Example:**

- $x_1$ in range [0, 100]
- $x_2$ in range [0, 100]
- $x_1 \cdot x_2$ in range [0, 10,000]

**Solutions:**

1. Scale original features before creating interactions
2. Scale interaction features after creation
3. Use models robust to different scales (trees)

---

## Polynomial Features

A special case of interactions plus powers:

**Degree 2 polynomial for features $x_1, x_2$:**

- Original: $x_1$, $x_2$
- Squared: $x_1^2$, $x_2^2$
- Interaction: $x_1 \cdot x_2$

This allows the model to fit:

$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_1^2 + \beta_4 x_2^2 + \beta_5 x_1 x_2
$$

---

## Interaction Features in Tree Models

Tree-based models (Random Forest, XGBoost) can learn interactions automatically:

**How trees capture interactions:**

A tree splits on $x_1$, then within each branch, splits on $x_2$. This naturally models the interaction $x_1 \cdot x_2$.

**When to still create interactions:**

- Shallow trees may miss complex interactions
- Explicit interactions can speed up learning
- Domain-specific interactions encode prior knowledge

---

## Interaction Features in Neural Networks

Neural networks learn interactions in hidden layers:

$$
h = \sigma(W_1 x_1 + W_2 x_2 + b)
$$

Non-linear activation allows learning of interaction effects.

**When to still create interactions:**

- Small networks may not capture complex interactions
- Explicit interactions can help with limited data
- Can improve training speed

---

## Feature Selection for Interactions

Not all interactions are useful. Selection methods:

**1. Correlation filtering:**

Keep interactions with high correlation to target.

**2. Statistical tests:**

Test if interaction coefficient is significantly different from zero.

**3. Regularization:**

Use L1 (Lasso) to automatically zero out unhelpful interactions.

**4. Tree-based importance:**

Use feature importance from trees to rank interactions.

---

## Automated Interaction Discovery

Let the algorithm find important interactions:

**1. Factorization Machines:**

Model pairwise interactions efficiently using latent factors.

**2. Deep and Cross Networks:**

Neural architecture that explicitly models feature crosses.

**3. Gradient Boosting:**

Sequential tree building naturally discovers interactions.

**4. Genetic Algorithms:**

Search over possible feature combinations.

---

## Common Interaction Patterns

**Synergy:** Combined effect greater than sum

$$
\text{Sales} = \beta_1 \cdot \text{Price} + \beta_2 \cdot \text{Advertising} + \beta_3 \cdot \text{Price} \cdot \text{Advertising}
$$

If $\beta_3 > 0$, advertising is more effective at lower prices.

**Substitution:** Combined effect less than sum

If $\beta_3 < 0$, the features partially substitute for each other.

---

## Interaction in Linear Regression

**Without interaction:**

$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2
$$

Effect of $x_1$ is constant: $\frac{\partial y}{\partial x_1} = \beta_1$

**With interaction:**

$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_1 x_2
$$

Effect of $x_1$ depends on $x_2$: $\frac{\partial y}{\partial x_1} = \beta_1 + \beta_3 x_2$

---

## Centering Before Interaction

To reduce correlation between main effects and interactions:

**1. Center features:**

$$
x_1' = x_1 - \bar{x_1}, \quad x_2' = x_2 - \bar{x_2}
$$

**2. Create interaction from centered features:**

$$
x_{int} = x_1' \cdot x_2'
$$

This makes coefficients more interpretable and reduces multicollinearity.

---

## Handling Zero Values

Division interactions have special cases:

**Problem:** $x_{new} = \frac{x_1}{x_2}$ is undefined when $x_2 = 0$

**Solutions:**

1. Add small constant: $x_{new} = \frac{x_1}{x_2 + \epsilon}$
2. Use indicator: Create separate feature for $x_2 = 0$ cases
3. Clip denominator: $x_{new} = \frac{x_1}{\max(x_2, \epsilon)}$

---

## Best Practices

**1. Start with domain knowledge:**

Create interactions that make business sense first.

**2. Validate with data:**

Ensure interactions improve model performance on holdout data.

**3. Be mindful of dimensionality:**

Do not create more features than you have samples.

**4. Document interactions:**

Keep track of what each interaction represents.

**5. Consider interpretability:**

Simple interactions (ratios, products) are easier to explain.

---

## Common Mistakes

**1. Creating too many interactions:**

Leads to overfitting and slow training.

**2. Not handling edge cases:**

Division by zero, log of negative numbers.

**3. Including redundant interactions:**

$x_1 \cdot x_2$ and $x_2 \cdot x_1$ are the same.

**4. Ignoring scale:**

Large interactions can dominate model training.

**5. Not validating:**

Assuming interactions help without testing.