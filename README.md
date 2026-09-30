# Loan Approval Predictor

Predicts the chance a loan application is approved and shows which factors helped or hurt.

## Run it
Double-click `index.html`. No install, server, or internet needed.

## Files
- `index.html` – page structure
- `style.css` – styling
- `model.js` – synthetic data generator + logistic regression (trained in the browser on load)
- `script.js` – form handling, gauge, and factor explanations

## How it works
1. `model.js` generates 5,000 synthetic applicants and labels them with a hidden lending rule plus noise.
2. It trains a logistic regression on 4,000 and tests on 1,000 (accuracy, precision, recall shown on the page).
3. Your inputs become 8 features (credit score, debt-to-income after the new installment, loan vs yearly income, years employed, degree, property, dependents, age).
4. The output is a probability; each feature's bar shows how much it moved the result versus an average applicant.

The data is synthetic, so this is for learning and demos only, not real lending decisions.

## Note
`index.html` already contains the CSS and JavaScript inside it, so it looks and works correctly even when opened on its own. `style.css`, `model.js` and `script.js` are the same code kept as separate files for reading and editing.
