import { useState } from "react";
import "./App.css";
import EnterDetailsForm from "./components/EnterDetailsForm";

export default function App() {
  const [result, setResult] = useState(null);

  return (
    <div className="container">
      <div className="header">
        <div className="brand">TrustBank</div>
        <div className="badge">Loan Eligibility • Demo Project</div>
      </div>

      <div className="grid">
        <div className="card">
          <h1>Check your loan eligibility</h1>
          <div className="sub">Enter your details below to get an instant decision.</div>
          <EnterDetailsForm onResult={setResult} />
        </div>

        <div className="card">
          <h2 style={{marginTop:0}}>Your result</h2>
          {!result && (
            <p className="mono">No result yet. Submit the form to see your decision.</p>
          )}

          {result && (
            <div className="result">
              <div className={`pill ${result.approved ? "ok" : "no"}`}>
                {result.approved ? "Approved!" : "Denied :("}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
