import { Routes, Route } from "react-router";
import Home from "./pages/Home/Home";
import Settings from './pages/Settings/Settings'
function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} key={"home"} />
      <Route path="/settings" element={<Settings />} key="settings" />
    </Routes>
  )
}

export default App
