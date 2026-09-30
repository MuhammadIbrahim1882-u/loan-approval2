/* Loan approval model: synthetic data + logistic regression trained in the browser. */
const LoanModel = (() => {
  const RATE = 0.10; // assumed yearly interest rate used to work out the monthly installment
  const FEATURES = [
    { key: 'credit', label: 'Credit score',            fmt: v => Math.round(v) },
    { key: 'dti',    label: 'Debt-to-income ratio',    fmt: v => (v * 100).toFixed(0) + '%' },
    { key: 'lti',    label: 'Loan size vs yearly income', fmt: v => v.toFixed(1) + '×' },
    { key: 'emp',    label: 'Years employed',          fmt: v => v.toFixed(0) + ' yrs' },
    { key: 'grad',   label: 'Graduate degree',         fmt: v => (v ? 'Yes' : 'No') },
    { key: 'prop',   label: 'Owns property',           fmt: v => (v ? 'Yes' : 'No') },
    { key: 'dep',    label: 'Dependents',              fmt: v => v.toFixed(0) },
    { key: 'age',    label: 'Age',                     fmt: v => v.toFixed(0) }
  ];

  function emi(loan, months) {
    const r = RATE / 12;
    return (loan * r) / (1 - Math.pow(1 + r, -months));
  }

  // Turns raw form values into the model's feature vector (order matches FEATURES).
  function toVector(a) {
    const payment = emi(a.loan, a.term);
    return [a.credit, (a.debt + payment) / a.income, a.loan / (a.income * 12),
            Math.min(a.emp, 20), a.grad, a.prop, a.dep, a.age];
  }

  function seededRandom(seed) {
    let s = seed >>> 0;
    return () => {
      s += 0x6D2B79F5;
      let t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function makeDataset(n, seed) {
    const rand = seededRandom(seed);
    const normal = () => Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());
    const terms = [12, 24, 36, 60, 84, 120, 180, 240];
    const X = [], y = [];
    for (let i = 0; i < n; i++) {
      const income = Math.exp(Math.log(4000) + 0.5 * normal());
      const age = 21 + Math.floor(rand() * 45);
      const a = {
        income,
        debt: income * rand() * rand() * 0.6,
        loan: income * 12 * (0.15 + rand() * 4),
        term: terms[Math.floor(rand() * terms.length)],
        credit: Math.min(850, Math.max(300, 660 + 90 * normal())),
        emp: Math.min(age - 20, rand() * 18),
        grad: rand() < 0.45 ? 1 : 0,
        prop: rand() < 0.35 ? 1 : 0,
        dep: Math.floor(rand() * 5),
        age
      };
      const v = toVector(a);
      // Hidden "bank policy" that generates the labels, plus noise so the data is not perfectly separable.
      const z = 0.012 * (v[0] - 650) - 6 * (v[1] - 0.40) - 0.7 * (v[2] - 2) + 0.10 * v[3]
              + 0.5 * v[4] + 0.6 * v[5] - 0.15 * v[6] + 0.02 * (Math.min(v[7], 50) - 30) + 0.5 * normal();
      X.push(v);
      y.push(z > 0 ? 1 : 0);
    }
    return { X, y };
  }

  const sigmoid = z => 1 / (1 + Math.exp(-z));

  function train() {
    const { X, y } = makeDataset(5000, 42);
    const split = 4000;
    const Xtr = X.slice(0, split), ytr = y.slice(0, split);
    const Xte = X.slice(split), yte = y.slice(split);
    const d = FEATURES.length;

    const mean = Array(d).fill(0), std = Array(d).fill(0);
    Xtr.forEach(r => r.forEach((v, j) => (mean[j] += v / split)));
    Xtr.forEach(r => r.forEach((v, j) => (std[j] += (v - mean[j]) ** 2 / split)));
    for (let j = 0; j < d; j++) std[j] = Math.sqrt(std[j]) || 1;
    const scale = r => r.map((v, j) => (v - mean[j]) / std[j]);

    const Ztr = Xtr.map(scale);
    let w = Array(d).fill(0), b = 0;
    const lr = 0.5, l2 = 0.001;
    for (let epoch = 0; epoch < 600; epoch++) {
      const gw = Array(d).fill(0); let gb = 0;
      for (let i = 0; i < split; i++) {
        let z = b; for (let j = 0; j < d; j++) z += w[j] * Ztr[i][j];
        const err = sigmoid(z) - ytr[i];
        for (let j = 0; j < d; j++) gw[j] += err * Ztr[i][j];
        gb += err;
      }
      for (let j = 0; j < d; j++) w[j] -= lr * (gw[j] / split + l2 * w[j]);
      b -= lr * gb / split;
    }

    function predict(vec) {
      const zs = scale(vec);
      const contribs = zs.map((v, j) => w[j] * v);
      const logit = b + contribs.reduce((s, c) => s + c, 0);
      return { prob: sigmoid(logit), contribs };
    }

    let tp = 0, tn = 0, fp = 0, fn = 0;
    Xte.forEach((r, i) => {
      const pred = predict(r).prob >= 0.5 ? 1 : 0;
      if (pred && yte[i]) tp++; else if (!pred && !yte[i]) tn++;
      else if (pred && !yte[i]) fp++; else fn++;
    });
    const total = tp + tn + fp + fn;
    const stats = {
      trainSize: split, testSize: total,
      accuracy: (tp + tn) / total,
      precision: tp / (tp + fp || 1),
      recall: tp / (tp + fn || 1),
      approvalRate: y.reduce((s, v) => s + v, 0) / y.length,
      confusion: { tp, tn, fp, fn }
    };
    return { predict, stats };
  }

  return { FEATURES, emi, toVector, train };
})();

if (typeof module !== 'undefined') module.exports = LoanModel;
