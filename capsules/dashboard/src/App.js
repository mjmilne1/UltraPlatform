import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { LineChart, Line, PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const API_URL = 'http://localhost:8000';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

function App() {
  const [capsules, setCapsules] = useState([]);
  const [selectedCapsule, setSelectedCapsule] = useState(null);
  const [transactions, setTransactions] = useState([]);
  const [allocations, setAllocations] = useState([]);
  const [performance, setPerformance] = useState(null);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadCapsules();
  }, []);

  const loadCapsules = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/v1/capsules`);
      setCapsules(response.data);
      if (response.data.length > 0) {
        loadCapsuleDetails(response.data[0].id);
      }
      setLoading(false);
    } catch (error) {
      console.error('Error loading capsules:', error);
      setLoading(false);
    }
  };

  const loadCapsuleDetails = async (capsuleId) => {
    try {
      const [capsule, trans, allocs, perf] = await Promise.all([
        axios.get(`${API_URL}/api/v1/capsules/${capsuleId}`),
        axios.get(`${API_URL}/api/v1/capsules/${capsuleId}/transactions`),
        axios.get(`${API_URL}/api/v1/capsules/${capsuleId}/allocations`),
        axios.get(`${API_URL}/api/v1/capsules/${capsuleId}/performance`).catch(() => ({ data: null }))
      ]);
      setSelectedCapsule(capsule.data);
      setTransactions(trans.data);
      setAllocations(allocs.data);
      setPerformance(perf.data);
    } catch (error) {
      console.error('Error loading capsule details:', error);
    }
  };

  const createCapsule = async (formData) => {
    try {
      await axios.post(`${API_URL}/api/v1/capsules`, formData);
      loadCapsules();
      setShowCreateForm(false);
    } catch (error) {
      console.error('Error creating capsule:', error);
    }
  };

  const addTransaction = async (capsuleId, transData) => {
    try {
      await axios.post(`${API_URL}/api/v1/capsules/${capsuleId}/transactions`, transData);
      loadCapsuleDetails(capsuleId);
    } catch (error) {
      console.error('Error adding transaction:', error);
    }
  };

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(value);
  };

  if (loading) {
    return <div style={styles.loading}>Loading Capsules Platform...</div>;
  }

  return (
    <div style={styles.app}>
      {/* Header */}
      <header style={styles.header}>
        <h1 style={styles.title}>💼 Capsules Platform</h1>
        <button style={styles.createButton} onClick={() => setShowCreateForm(true)}>
          + New Capsule
        </button>
      </header>

      {/* Main Content */}
      <div style={styles.container}>
        {/* Sidebar */}
        <aside style={styles.sidebar}>
          <h2 style={styles.sidebarTitle}>Capsules ({capsules.length})</h2>
          {capsules.map(capsule => (
            <div
              key={capsule.id}
              style={{
                ...styles.capsuleCard,
                ...(selectedCapsule?.id === capsule.id ? styles.capsuleCardActive : {})
              }}
              onClick={() => loadCapsuleDetails(capsule.id)}
            >
              <div style={styles.capsuleType}>{capsule.capsule_type}</div>
              <div style={styles.capsuleGoal}>{formatCurrency(capsule.goal_amount)}</div>
              <div style={styles.capsuleValue}>Current: {formatCurrency(capsule.current_value)}</div>
            </div>
          ))}
        </aside>

        {/* Main Panel */}
        <main style={styles.main}>
          {selectedCapsule ? (
            <>
              {/* Overview Cards */}
              <div style={styles.cardGrid}>
                <div style={styles.card}>
                  <div style={styles.cardTitle}>Current Value</div>
                  <div style={styles.cardValue}>{formatCurrency(selectedCapsule.current_value)}</div>
                  <div style={styles.cardSubtext}>Goal: {formatCurrency(selectedCapsule.goal_amount)}</div>
                </div>
                <div style={styles.card}>
                  <div style={styles.cardTitle}>Progress</div>
                  <div style={styles.cardValue}>
                    {((selectedCapsule.current_value / selectedCapsule.goal_amount) * 100).toFixed(1)}%
                  </div>
                  <div style={styles.cardSubtext}>Target: {selectedCapsule.goal_date}</div>
                </div>
                <div style={styles.card}>
                  <div style={styles.cardTitle}>Performance</div>
                  <div style={styles.cardValue}>
                    {performance?.return_percentage ? `${performance.return_percentage.toFixed(2)}%` : 'N/A'}
                  </div>
                  <div style={styles.cardSubtext}>Total Return</div>
                </div>
                <div style={styles.card}>
                  <div style={styles.cardTitle}>Transactions</div>
                  <div style={styles.cardValue}>{transactions.length}</div>
                  <div style={styles.cardSubtext}>All Time</div>
                </div>
              </div>

              {/* Allocations Chart */}
              {allocations.length > 0 && (
                <div style={styles.chartCard}>
                  <h3 style={styles.chartTitle}>Portfolio Allocation</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={allocations}
                        dataKey="current_percentage"
                        nameKey="asset_class"
                        cx="50%"
                        cy="50%"
                        outerRadius={100}
                        label
                      >
                        {allocations.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                  <div style={styles.allocationList}>
                    {allocations.map((alloc, idx) => (
                      <div key={idx} style={styles.allocationItem}>
                        <span style={{...styles.allocationDot, backgroundColor: COLORS[idx % COLORS.length]}}></span>
                        <span style={styles.allocationName}>{alloc.asset_class}</span>
                        <span style={styles.allocationPercent}>{alloc.current_percentage}%</span>
                        <span style={styles.allocationValue}>{formatCurrency(alloc.current_value)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Transactions */}
              <div style={styles.chartCard}>
                <h3 style={styles.chartTitle}>Recent Transactions</h3>
                <div style={styles.transactionList}>
                  {transactions.slice(0, 10).map(trans => (
                    <div key={trans.id} style={styles.transactionItem}>
                      <div style={styles.transactionType}>{trans.transaction_type}</div>
                      <div style={styles.transactionDesc}>{trans.description}</div>
                      <div style={{
                        ...styles.transactionAmount,
                        color: trans.transaction_type === 'withdrawal' ? '#f44336' : '#4caf50'
                      }}>
                        {trans.transaction_type === 'withdrawal' ? '-' : '+'}
                        {formatCurrency(Math.abs(trans.amount))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Quick Actions */}
              <div style={styles.actions}>
                <button 
                  style={styles.actionButton}
                  onClick={() => {
                    const amount = prompt('Enter deposit amount:');
                    if (amount) {
                      addTransaction(selectedCapsule.id, {
                        transaction_type: 'deposit',
                        amount: parseFloat(amount),
                        description: 'Manual deposit'
                      });
                    }
                  }}
                >
                  💰 Add Deposit
                </button>
                <button 
                  style={styles.actionButton}
                  onClick={async () => {
                    try {
                      await axios.post(`${API_URL}/api/v1/capsules/${selectedCapsule.id}/rebalance`);
                      alert('Portfolio rebalanced successfully!');
                      loadCapsuleDetails(selectedCapsule.id);
                    } catch (error) {
                      alert('Error rebalancing: ' + error.message);
                    }
                  }}
                >
                  ⚖️ Rebalance
                </button>
              </div>
            </>
          ) : (
            <div style={styles.empty}>
              <h2>No Capsule Selected</h2>
              <p>Select a capsule from the sidebar or create a new one</p>
            </div>
          )}
        </main>
      </div>

      {/* Create Modal */}
      {showCreateForm && (
        <CreateCapsuleModal 
          onClose={() => setShowCreateForm(false)}
          onCreate={createCapsule}
        />
      )}
    </div>
  );
}

function CreateCapsuleModal({ onClose, onCreate }) {
  const [formData, setFormData] = useState({
    client_id: '',
    capsule_type: 'retirement',
    goal_amount: '',
    goal_date: ''
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    onCreate({
      ...formData,
      goal_amount: parseFloat(formData.goal_amount)
    });
  };

  return (
    <div style={styles.modal} onClick={onClose}>
      <div style={styles.modalContent} onClick={(e) => e.stopPropagation()}>
        <h2 style={styles.modalTitle}>Create New Capsule</h2>
        <form onSubmit={handleSubmit} style={styles.form}>
          <div style={styles.formGroup}>
            <label style={styles.label}>Client ID</label>
            <input
              style={styles.input}
              type="text"
              value={formData.client_id}
              onChange={(e) => setFormData({...formData, client_id: e.target.value})}
              required
            />
          </div>
          <div style={styles.formGroup}>
            <label style={styles.label}>Type</label>
            <select
              style={styles.input}
              value={formData.capsule_type}
              onChange={(e) => setFormData({...formData, capsule_type: e.target.value})}
            >
              <option value="retirement">Retirement</option>
              <option value="education">Education</option>
              <option value="home_deposit">Home Deposit</option>
              <option value="emergency">Emergency Fund</option>
            </select>
          </div>
          <div style={styles.formGroup}>
            <label style={styles.label}>Goal Amount</label>
            <input
              style={styles.input}
              type="number"
              value={formData.goal_amount}
              onChange={(e) => setFormData({...formData, goal_amount: e.target.value})}
              required
            />
          </div>
          <div style={styles.formGroup}>
            <label style={styles.label}>Goal Date</label>
            <input
              style={styles.input}
              type="date"
              value={formData.goal_date}
              onChange={(e) => setFormData({...formData, goal_date: e.target.value})}
              required
            />
          </div>
          <div style={styles.modalActions}>
            <button type="button" style={styles.cancelButton} onClick={onClose}>Cancel</button>
            <button type="submit" style={styles.submitButton}>Create Capsule</button>
          </div>
        </form>
      </div>
    </div>
  );
}

const styles = {
  app: { minHeight: '100vh', backgroundColor: '#f5f5f5' },
  header: { 
    backgroundColor: '#1976d2', 
    color: 'white', 
    padding: '20px 40px', 
    display: 'flex', 
    justifyContent: 'space-between', 
    alignItems: 'center',
    boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
  },
  title: { margin: 0, fontSize: '24px' },
  createButton: { 
    backgroundColor: '#4caf50', 
    color: 'white', 
    border: 'none', 
    padding: '10px 20px', 
    borderRadius: '4px', 
    cursor: 'pointer',
    fontSize: '14px',
    fontWeight: 'bold'
  },
  container: { display: 'flex', height: 'calc(100vh - 80px)' },
  sidebar: { 
    width: '300px', 
    backgroundColor: 'white', 
    padding: '20px', 
    overflowY: 'auto',
    borderRight: '1px solid #e0e0e0'
  },
  sidebarTitle: { marginBottom: '20px', fontSize: '18px', color: '#333' },
  capsuleCard: { 
    padding: '15px', 
    marginBottom: '10px', 
    backgroundColor: '#f9f9f9', 
    borderRadius: '8px', 
    cursor: 'pointer',
    border: '2px solid transparent',
    transition: 'all 0.2s'
  },
  capsuleCardActive: { 
    backgroundColor: '#e3f2fd', 
    border: '2px solid #1976d2'
  },
  capsuleType: { fontWeight: 'bold', marginBottom: '5px', textTransform: 'capitalize' },
  capsuleGoal: { fontSize: '18px', color: '#1976d2', marginBottom: '5px' },
  capsuleValue: { fontSize: '12px', color: '#666' },
  main: { flex: 1, padding: '20px', overflowY: 'auto' },
  cardGrid: { 
    display: 'grid', 
    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
    gap: '20px', 
    marginBottom: '20px'
  },
  card: { 
    backgroundColor: 'white', 
    padding: '20px', 
    borderRadius: '8px',
    boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
  },
  cardTitle: { fontSize: '14px', color: '#666', marginBottom: '10px' },
  cardValue: { fontSize: '28px', fontWeight: 'bold', color: '#333', marginBottom: '5px' },
  cardSubtext: { fontSize: '12px', color: '#999' },
  chartCard: { 
    backgroundColor: 'white', 
    padding: '20px', 
    borderRadius: '8px', 
    marginBottom: '20px',
    boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
  },
  chartTitle: { marginBottom: '20px', color: '#333' },
  allocationList: { marginTop: '20px' },
  allocationItem: { 
    display: 'flex', 
    alignItems: 'center', 
    padding: '10px 0', 
    borderBottom: '1px solid #f0f0f0'
  },
  allocationDot: { 
    width: '12px', 
    height: '12px', 
    borderRadius: '50%', 
    marginRight: '10px'
  },
  allocationName: { flex: 1, textTransform: 'capitalize' },
  allocationPercent: { marginRight: '20px', fontWeight: 'bold' },
  allocationValue: { color: '#666' },
  transactionList: {},
  transactionItem: { 
    display: 'flex', 
    padding: '15px 0', 
    borderBottom: '1px solid #f0f0f0'
  },
  transactionType: { 
    width: '120px', 
    textTransform: 'capitalize', 
    fontWeight: 'bold'
  },
  transactionDesc: { flex: 1, color: '#666' },
  transactionAmount: { fontWeight: 'bold', fontSize: '16px' },
  actions: { display: 'flex', gap: '10px' },
  actionButton: { 
    backgroundColor: '#1976d2', 
    color: 'white', 
    border: 'none', 
    padding: '12px 24px', 
    borderRadius: '4px', 
    cursor: 'pointer',
    fontSize: '14px',
    fontWeight: 'bold'
  },
  empty: { 
    textAlign: 'center', 
    padding: '60px', 
    color: '#999'
  },
  loading: { 
    display: 'flex', 
    justifyContent: 'center', 
    alignItems: 'center', 
    height: '100vh', 
    fontSize: '20px', 
    color: '#666'
  },
  modal: { 
    position: 'fixed', 
    top: 0, 
    left: 0, 
    right: 0, 
    bottom: 0, 
    backgroundColor: 'rgba(0,0,0,0.5)', 
    display: 'flex', 
    justifyContent: 'center', 
    alignItems: 'center',
    zIndex: 1000
  },
  modalContent: { 
    backgroundColor: 'white', 
    padding: '30px', 
    borderRadius: '8px', 
    width: '500px',
    maxWidth: '90%'
  },
  modalTitle: { marginBottom: '20px', color: '#333' },
  form: {},
  formGroup: { marginBottom: '20px' },
  label: { display: 'block', marginBottom: '5px', fontWeight: 'bold', color: '#333' },
  input: { 
    width: '100%', 
    padding: '10px', 
    border: '1px solid #ddd', 
    borderRadius: '4px',
    fontSize: '14px'
  },
  modalActions: { display: 'flex', justifyContent: 'flex-end', gap: '10px', marginTop: '20px' },
  cancelButton: { 
    padding: '10px 20px', 
    border: '1px solid #ddd', 
    backgroundColor: 'white', 
    borderRadius: '4px', 
    cursor: 'pointer'
  },
  submitButton: { 
    padding: '10px 20px', 
    backgroundColor: '#4caf50', 
    color: 'white', 
    border: 'none', 
    borderRadius: '4px', 
    cursor: 'pointer',
    fontWeight: 'bold'
  }
};

export default App;
