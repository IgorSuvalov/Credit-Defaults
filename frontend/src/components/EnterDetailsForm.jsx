import React, { useState } from "react";
import api from "../api";

const isEmptyOrNotNumber = (v) => v === "" || Number.isNaN(Number(v));

export default function EnterDetailsForm({ onResult }) {
  const [age, setAge] = useState("");
  const [income, setIncome] = useState("");
  const [homeOwnership, setHomeOwnership] = useState("rent");
  const [empLength, setEmpLength] = useState("");
  const [loanAmount, setLoanAmount] = useState("");
  const [defOnFile, setDefOnFile] = useState("no");
  const [loanIntent, setLoanIntent] = useState("personal");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  function validate() {
    if ([age, income, empLength, loanAmount].some(isEmptyOrNotNumber)) {
      setError("Please fill all numeric fields with valid numbers.");
      return false;
    }
    if (Number(age) <= 0 || Number(loanAmount) <= 0 || Number(income) < 0 || Number(empLength) < 0) {
      setError("Age and loan amount must be > 0; income and employment length must be ≥ 0.");
      return false;
    }
    setError(null);
    return true;
  }

  async function handleSubmit(e) {
    e.preventDefault();
    if (!validate()) return;

    const payload = {
      // IMPORTANT: match FastAPI field names
      age: Number(age),
      income: Number(income),
      home_ownership: homeOwnership,
      employment_length: Number(empLength),
      loan_amount: Number(loanAmount),
      def_on_file: defOnFile === "yes" ? 1 : 0,
      loan_intent: loanIntent,
    };

    console.log("Payload being sent:", payload);

    setLoading(true);
    try {
      const res = await api.post("/score", payload);
      const data = res.data ?? {};
      if (typeof onResult === "function") onResult(data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <form className="form" onSubmit={handleSubmit}>
      <div className="form-grid">
        <div>
          <label className="label">Age</label>
          <input className="input" type="number" min="0" max="120" value={age} onChange={(e)=>setAge(e.target.value)} required />
        </div>
        <div>
          <label className="label">Annual Income</label>
          <input className="input" type="number" min="0" max="100000000" value={income}
                 onChange={(e)=>setIncome(e.target.value)} required />
        </div>
        <div>
          <label className="label">Home Ownership</label>
          <select className="select" value={homeOwnership} onChange={(e)=>setHomeOwnership(e.target.value)}>
            <option value="rent">Rent</option>
            <option value="mortgage">Mortgage</option>
            <option value="own">Own</option>
            <option value="other">Other</option>
          </select>
        </div>
        <div>
          <label className="label">Employment Length (years)</label>
          <input className="input" type="number" min="0" max="110" value={empLength} onChange={(e)=>setEmpLength(e.target.value)}
                 onInvalid={(e) => {
                   const target = e.target;
                   if (target.value.rangeOverflow) {target.setCustomValidity("Employment length seems too high. Please enter a realistic number.");
                   } else if (target.value.rangeUnderflow) {target.setCustomValidity("Employment length cannot be negative.");
                   } else if (target.value.valueMissing) {target.setCustomValidity("Please fill out this field.");}
                 }}
                 onInput={(e) => e.target.setCustomValidity("")} required />
        </div>
        <div>
          <label className="label">Loan Amount</label>
          <input className="input" type="number" min="1" max="1000000000" value={loanAmount}
                 onChange={(e)=>setLoanAmount(e.target.value)} required />
        </div>
        <div>
          <label className="label">Default on File</label>
          <select className="select" value={defOnFile} onChange={(e)=>setDefOnFile(e.target.value)}>
            <option value="no">No</option>
            <option value="yes">Yes</option>
          </select>
        </div>
        <div style={{ gridColumn: "1 / -1" }}>
          <label className="label">Loan Intent</label>
          <select className="select" value={loanIntent} onChange={(e)=>setLoanIntent(e.target.value)}>
            <option value="debtconsolidation">Debt Consolidation</option>
            <option value="personal">Personal</option>
            <option value="education">Education</option>
            <option value="medical">Medical</option>
            <option value="venture">Venture</option>
            <option value="homeimprovement">Home Improvement</option>
          </select>
        </div>
      </div>

      {error && <div className="error">{error}</div>}

      <div className="row" style={{ marginTop: 14 }}>
        <button className="btn" type="submit" disabled={loading}>
          {loading ? "Checking…" : "Check eligibility"}
        </button>
      </div>
    </form>
  );
}