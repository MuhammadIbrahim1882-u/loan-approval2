const { predict, stats } = LoanModel.train();
const $ = id => document.getElementById(id);
const ids = ['income', 'debt', 'loan', 'term', 'credit', 'emp', 'age', 'dep', 'grad', 'prop'];
const ARC_LEN = Math.PI * 80;

const PRESETS = {
  strong:     { income: 8000, debt: 300,  loan: 40000,  term: 60,  credit: 780, emp: 10, age: 40, dep: 1, grad: 1, prop: 1 },
  borderline: { income: 4000, debt: 700,  loan: 70000,  term: 84,  credit: 650, emp: 3,  age: 30, dep: 2, grad: 0, prop: 0 },
  weak:       { income: 2500, debt: 1000, loan: 150000, term: 60,  credit: 520, emp: 1,  age: 24, dep: 3, grad: 0, prop: 0 }
};

const money = v => Math.round(v).toLocaleString('en-US');

function read() {
  const o = {};
  ids.forEach(id => (o[id] = parseFloat($(id).value)));
  return o;
}

function validate(a) {
  if (!(a.income > 0)) return 'Enter a monthly income above 0.';
  if (!(a.loan > 0)) return 'Enter a loan amount above 0.';
  if (ids.some(id => Number.isNaN(a[id]))) return 'Fill in every field to see a result.';
  if (a.debt < 0 || a.emp < 0 || a.dep < 0) return 'Values cannot be negative.';
  return '';
}

function update() {
  $('creditOut').textContent = $('credit').value;
  const a = read();
  const problem = validate(a);
  $('error').hidden = !problem;
  $('error').textContent = problem;
  if (problem) return;

  const vec = LoanModel.toVector(a);
  const { prob, contribs } = predict(vec);
  const pct = Math.round(prob * 100);

  $('arc').style.strokeDashoffset = ARC_LEN * (1 - prob);
  $('pct').textContent = pct + '%';

  let verdict, cls;
  if (prob >= 0.65)      { verdict = 'Likely to be approved'; cls = 'good'; }
  else if (prob >= 0.40) { verdict = 'Borderline: could go either way'; cls = 'mid'; }
  else                   { verdict = 'Likely to be declined'; cls = 'bad'; }
  $('verdict').textContent = verdict;
  $('verdict').className = 'verdict ' + cls;
  $('arc').setAttribute('class', 'arc ' + cls);

  $('emi').textContent = 'Estimated installment: ' + money(LoanModel.emi(a.loan, a.term)) +
    ' per month (at 10% yearly interest)';

  const max = Math.max(...contribs.map(Math.abs), 0.01);
  const rows = LoanModel.FEATURES.map((f, i) => ({ f, c: contribs[i], v: vec[i] }))
    .sort((x, y) => Math.abs(y.c) - Math.abs(x.c));
  $('factors').innerHTML = rows.map(({ f, c, v }) => {
    const w = (Math.abs(c) / max) * 50;
    const side = c >= 0 ? 'pos' : 'neg';
    return `<li><div class="fl"><span>${f.label}</span><em>${f.fmt(v)}</em></div>
      <div class="bar"><i class="${side}" style="width:${w}%"></i></div></li>`;
  }).join('');
}

function applyPreset(name) {
  const p = PRESETS[name];
  ids.forEach(id => ($(id).value = p[id]));
  update();
}

document.querySelectorAll('[data-preset]').forEach(b =>
  b.addEventListener('click', () => applyPreset(b.dataset.preset)));
ids.forEach(id => $(id).addEventListener('input', update));

$('trainSize').textContent = stats.trainSize.toLocaleString('en-US');
$('testSize').textContent = stats.testSize.toLocaleString('en-US');
$('acc').textContent = (stats.accuracy * 100).toFixed(1) + '%';
$('prec').textContent = (stats.precision * 100).toFixed(1) + '%';
$('rec').textContent = (stats.recall * 100).toFixed(1) + '%';
$('rate').textContent = (stats.approvalRate * 100).toFixed(0) + '%';

$('arc').style.strokeDasharray = ARC_LEN;
update();
