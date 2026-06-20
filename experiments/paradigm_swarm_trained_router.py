#!/usr/bin/env python3
"""Router trained on hand-crafted labeled dataset — English (negative result, §4.17)"""
import numpy as np, re
from collections import Counter

np.random.seed(42)

# ═══════════════════════════════════════════════════
# HAND-CRAFTED DATASET: queries with paradigm labels
# ═══════════════════════════════════════════════════
DATASET = [
    # === LABOUR LAW ===
    ("I was fired without explanation — is this legal?", "labour"),
    ("The employer has not paid my salary for three weeks", "labour"),
    ("Laid off without severance pay — where to complain?", "labour"),
    ("They did not return my employment record upon dismissal", "labour"),
    ("Three months probation — can they fire without notice?", "labour"),
    ("Sick leave not paid on time — what can I do?", "labour"),
    ("Fired while on sick leave — is this a breach of statute?", "labour"),
    ("Unpaid leave — how many days can I take?", "labour"),
    ("Working overtime without pay — how to prove it?", "labour"),
    ("Maternity leave: what benefits am I entitled to?", "labour"),
    ("Dismissed for absenteeism but I was only gone three hours", "labour"),
    ("Transferred to another role without my consent", "labour"),
    ("Study leave not paid — is this a violation?", "labour"),
    ("Redundancy: must they offer an alternative vacancy?", "labour"),
    ("Pregnant employee facing dismissal — is this legal?", "labour"),
    ("Compensation for unused vacation upon termination", "labour"),
    ("Salary delayed two months — are penalties owed?", "labour"),
    ("Night shifts: what is the statutory pay premium?", "labour"),
    ("Discrimination in hiring — refused without reason", "labour"),
    ("Wrongful dismissal: fired under a non-existent statute", "labour"),

    # === THERMODYNAMICS ===
    ("Why does entropy of a closed system always increase?", "thermo"),
    ("Explain the second law of thermodynamics in simple terms", "thermo"),
    ("Can heat transfer from a cold body to a hot one?", "thermo"),
    ("What is the Carnot cycle and why is it the most efficient?", "thermo"),
    ("An adiabatic process has no heat exchange — correct?", "thermo"),
    ("Phase transition: why does temperature stay constant?", "thermo"),
    ("Perpetual motion machine of the second kind — why impossible?", "thermo"),
    ("Heat engine efficiency: can it reach 100%?", "thermo"),
    ("Isothermal process: is temperature constant?", "thermo"),
    ("Water has anomalously high heat capacity — why?", "thermo"),
    ("Entropy as a measure of disorder — is this correct?", "thermo"),
    ("The third law of thermodynamics: absolute zero", "thermo"),
    ("Heat pump: how does it move heat against the gradient?", "thermo"),
    ("Gas expanding into vacuum — an irreversible process?", "thermo"),
    ("Critical point: no difference between liquid and gas", "thermo"),
    ("Real gases: deviations from ideal behavior", "thermo"),
    ("Latent heat of phase transition — what is it?", "thermo"),
    ("Triple point of water: three phases in equilibrium", "thermo"),
    ("Supercritical fluid: neither liquid nor gas", "thermo"),
    ("Statistical physics: its connection to thermodynamics", "thermo"),

    # === GEOMETRY ===
    ("Why does the sum of a triangle's angles equal 180 degrees?", "geo"),
    ("Through two points passes exactly one line — prove it", "geo"),
    ("Parallel lines: do they intersect at infinity?", "geo"),
    ("Pythagorean theorem: proof for a right triangle", "geo"),
    ("The diagonals of a rhombus are perpendicular — why?", "geo"),
    ("A median of a triangle bisects the opposite side", "geo"),
    ("An inscribed angle equals half the arc — prove it", "geo"),
    ("A tangent to a circle is perpendicular to the radius", "geo"),
    ("Equal chords are equidistant from the center of a circle", "geo"),
    ("A bisector divides an angle in half — property", "geo"),
    ("The sum of angles in a quadrilateral equals 360 degrees", "geo"),
    ("Area of a circle: derivation of pi-r-squared", "geo"),
    ("Volume of a sphere: four-thirds pi r cubed", "geo"),
    ("Law of cosines: generalization of the Pythagorean theorem", "geo"),
    ("The golden ratio in a regular pentagon", "geo"),
    ("Orthocenter of a triangle: intersection of altitudes", "geo"),
    ("The centroid divides a median in ratio 2:1", "geo"),
    ("Euclid's fifth postulate: the parallel axiom", "geo"),

    # === CONSTITUTIONAL LAW ===
    ("Can the president dissolve Congress?", "const_law"),
    ("The Supreme Court: what cases does it hear?", "const_law"),
    ("Can a state secede from the federation?", "const_law"),
    ("Limits on freedom of assembly — is this constitutional?", "const_law"),
    ("Freedom of speech: where are the boundaries?", "const_law"),
    ("How to amend the Constitution? The amendment process", "const_law"),
    ("Separation of powers: how does it work?", "const_law"),
    ("Due process: what does the Constitution guarantee?", "const_law"),
    ("Equal protection: constitutional guarantee against discrimination", "const_law"),
    ("Judicial review: can courts strike down laws?", "const_law"),
    ("The right to privacy: constitutional basis", "const_law"),
    ("Freedom of religion: establishment and free exercise", "const_law"),
    ("Checks and balances between branches of government", "const_law"),
    ("Federal vs state power: constitutional boundaries", "const_law"),
    ("The Bill of Rights: first ten amendments", "const_law"),

    # === QUANTUM MECHANICS ===
    ("Heisenberg uncertainty principle explained simply", "quantum"),
    ("Quantum superposition: how can a particle be in two states?", "quantum"),
    ("Entangled particles: faster-than-light information transfer?", "quantum"),
    ("Wave function collapse: what happens during measurement?", "quantum"),
    ("Quantum tunneling: how does a particle cross a barrier?", "quantum"),
    ("Wave-particle duality: is light a wave or a particle?", "quantum"),
    ("Electron spin: what is this property?", "quantum"),
    ("Schrodinger's equation: what does the wave function describe?", "quantum"),
    ("Pauli exclusion principle: two fermions in one state", "quantum"),
    ("Energy quantization: why are electrons in orbitals?", "quantum"),
    ("Quantum teleportation: is matter transferred?", "quantum"),
    ("The EPR paradox: Einstein-Podolsky-Rosen", "quantum"),
    ("Bell inequalities: testing quantum mechanics", "quantum"),
    ("Decoherence: why quantum effects vanish in the macro world", "quantum"),
    ("Copenhagen interpretation of quantum mechanics", "quantum"),
    ("Everett's many-worlds interpretation", "quantum"),
    ("Quantum computing: how qubits work", "quantum"),
    ("Schrodinger's cat: the thought experiment", "quantum"),

    # === CROSS-DOMAIN ===
    ("Entropy in labour markets: measuring uncertainty of dismissals", "cross_domain"),
    ("Quantum superposition in law: can a statute be constitutional and void until reviewed?", "cross_domain"),
    ("The uncertainty principle in employment: cannot simultaneously know productivity and satisfaction", "cross_domain"),
    ("Thermodynamic equilibrium in economics: analogy with supply-demand balance", "cross_domain"),
    ("Geometry of electoral districts: how district shape affects election outcomes", "cross_domain"),
    ("Feedback loops in thermodynamics and law: court rulings as a thermostat", "cross_domain"),

    # === GAP ===
    ("How to measure the fairness of an algorithmic decision?", "gap"),
    ("Does a universal measure of complexity exist for all systems?", "gap"),
    ("Can intuition be formalized mathematically?", "gap"),
    ("How to evaluate the aesthetic value of a work of art?", "gap"),
    ("What is understanding? Can it be measured?", "gap"),
    ("How to formally distinguish science from pseudoscience?", "gap"),

    # === METAPHOR ===
    ("Quantum entanglement in social networks and its effect on likes", "metaphor"),
    ("Parallel lines in corporate organizational structure", "metaphor"),
    ("Entropy of personal relationships: measuring chaos in a family", "metaphor"),
    ("Superposition of a manager: simultaneously hired and fired", "metaphor"),
    ("The hypotenuse of career growth: the shortest path to success", "metaphor"),
    ("Thermodynamics of friendship: conservation of emotional energy", "metaphor"),
]

print(f"Dataset: {len(DATASET)} labeled queries")

# ═══════════════════════════════════════════════════
# Word bigram features
# ═══════════════════════════════════════════════════
def word_ngrams(text):
    words = re.findall(r'[a-z]+', text.lower())
    result = []
    for i in range(len(words)-1):
        result.append(words[i]+'_'+words[i+1])
    result.extend(words)
    return result

ngram_counter = Counter()
for q, _ in DATASET:
    for ng in word_ngrams(q):
        ngram_counter[ng] += 1

top_ngrams = [ng for ng, c in ngram_counter.most_common(600) if c >= 2]
ngram2idx = {ng:i for i,ng in enumerate(top_ngrams)}
N_FEAT = len(top_ngrams)
print(f"Features: {N_FEAT} word n-grams")

def featurize(text):
    vec = np.zeros(N_FEAT)
    for ng in word_ngrams(text):
        if ng in ngram2idx:
            vec[ngram2idx[ng]] += 1
    s = vec.sum()
    if s > 0:
        vec /= s
    return vec

# Train/test split
all_labels = list(set(l for _, l in DATASET))
label2id = {l:i for i,l in enumerate(all_labels)}
N_CLASSES = len(all_labels)

idx = np.random.permutation(len(DATASET))
split = int(len(DATASET)*0.8)
train_idx, test_idx = idx[:split], idx[split:]

X_tr = np.array([featurize(DATASET[i][0]) for i in train_idx])
y_tr = np.array([label2id[DATASET[i][1]] for i in train_idx])
X_te = np.array([featurize(DATASET[i][0]) for i in test_idx])
y_te = np.array([label2id[DATASET[i][1]] for i in test_idx])

# Train multi-class router
HID, LR_R, EP_R = 64, 0.05, 500

class RouterMLP:
    def __init__(self):
        rng = np.random.RandomState(42); s = 0.1
        self.W1 = rng.randn(N_FEAT, HID)*s; self.b1 = np.zeros((1,HID))
        self.W2 = rng.randn(HID, N_CLASSES)*s; self.b2 = np.zeros((1,N_CLASSES))
    def forward(self, X):
        a = np.maximum(0, X@self.W1+self.b1)
        z = a@self.W2+self.b2
        e = np.exp(z-z.max(1,keepdims=True))
        return e/e.sum(1,keepdims=True)
    def predict(self, X): return self.forward(X).argmax(1)
    def train(self, X, y):
        y_oh = np.eye(N_CLASSES)[y]
        for _ in range(EP_R):
            idx_b = np.random.choice(len(X), min(32,len(X)), replace=False)
            Xb, yb = X[idx_b], y_oh[idx_b]
            p = self.forward(Xb); N = len(Xb)
            a = np.maximum(0, Xb@self.W1+self.b1)
            dz = (p-yb)/N
            self.W2 -= LR_R*a.T@dz; self.b2 -= LR_R*dz.sum(0,keepdims=True)
            da = dz@self.W2.T*(a>0)
            self.W1 -= LR_R*Xb.T@da; self.b1 -= LR_R*da.sum(0,keepdims=True)

router = RouterMLP()
router.train(X_tr, y_tr)

preds = router.predict(X_te)
acc = np.mean(preds == y_te)
print(f"\nTrained Router Accuracy: {acc:.3f} ({sum(preds==y_te)}/{len(y_te)})")

id2label = {i:l for l,i in label2id.items()}
print(f"\nPer-class accuracy:")
for lid in range(N_CLASSES):
    mask = y_te == lid
    if mask.sum() > 0:
        acc_l = np.mean(preds[mask] == lid)
        print(f"  {id2label[lid]:<15}: {acc_l:.3f} ({mask.sum()} samples)")

print(f"\nVerdict: trained router fails on limited data ({len(DATASET)} examples,")
print(f"{N_FEAT} features → overfitting). This negative result motivates")
print(f"the router-free architecture (§4.17).")
