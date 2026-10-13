import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { lazy, Suspense } from 'react'
import Layout from './components/Layout'
import OverviewPage from './pages/OverviewPage'
import DatasetExplorerPage from './pages/DatasetExplorerPage'
import ExperimentSetupPage from './pages/ExperimentSetupPage'
import ActiveLearningDashboardPage from './pages/ActiveLearningDashboardPage'
import CandidateExplorerPage from './pages/CandidateExplorerPage'
import SelectionExplainerPage from './pages/SelectionExplainerPage'
import ExperimentHistoryPage from './pages/ExperimentHistoryPage'
import StrategyComparisonPage from './pages/StrategyComparisonPage'
import MaterialDiscoveryPage from './pages/MaterialDiscoveryPage'
import BenchmarkLandingPage from './pages/BenchmarkLandingPage'
import ReportsPage from './pages/ReportsPage'
const ExplainabilityPage = lazy(() => import('./pages/ExplainabilityPage'))

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<OverviewPage />} />
          <Route path="dataset" element={<DatasetExplorerPage />} />
          <Route path="benchmark" element={<BenchmarkLandingPage />} />
          <Route path="reports" element={<ReportsPage />} />
          <Route path="setup" element={<ExperimentSetupPage />} />
          <Route path="history" element={<ExperimentHistoryPage />} />
          <Route path="experiment/:id" element={<ActiveLearningDashboardPage />} />
          <Route path="experiment/:id/candidates" element={<CandidateExplorerPage />} />
          <Route path="experiment/:id/candidates/:candidateId" element={<SelectionExplainerPage />} />
          <Route path="experiment/:id/explainability" element={<Suspense fallback={<div className="p-6 text-t3">Loading explainability tools…</div>}><ExplainabilityPage /></Suspense>} />
          <Route path="experiment/:id/history" element={<ExperimentHistoryPage />} />
          <Route path="experiment/:id/compare" element={<StrategyComparisonPage />} />
          <Route path="experiment/:id/discovery" element={<MaterialDiscoveryPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
