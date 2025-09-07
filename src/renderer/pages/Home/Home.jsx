import React, { useState, useEffect, useRef } from 'react';
import Navbar from '../../components/Navbar/Navbar';
import './Home.css';
import { useTheme } from '../../contexts/ThemeProvider';

const App = () => {
    const messagesEndRef = useRef(null);
    const { darkMode, setDarkMode } = useTheme();
    const [chatMessages, setChatMessages] = useState([
        { sender: 'ai', text: '[SYSTEM INIT] TESS v2.1 loaded. Technical Engine for System Support online. How may I assist you?' }
    ]);
    const [cpu, setCpu] = useState(45);
    const [ram, setRam] = useState(62);
    const [disk, setDisk] = useState(85);

    useEffect(() => {
        if (darkMode) {
            document.documentElement.classList.add('dark');
        } else {
            document.documentElement.classList.remove('dark');
        }
    }, [darkMode]);

    const handleChatSubmit = (e) => {
        e.preventDefault();
        const input = e.target.elements['chat-input'];
        const message = input.value.trim().toUpperCase();
        if (message === '') return;
        setChatMessages(prevMessages => [...prevMessages, { sender: 'user', text: message }]);
        input.value = '';
    };


    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [chatMessages]);

    useEffect(() => {
        const unsub = window.electronAPI.subscribeDeviceResourceUsage((data) => {
            setCpu((Math.floor(data.cpuUsage * 100)));
            setRam((Math.floor(data.ramUsage * 100)));
            setDisk((Math.floor(data.diskUsage * 100)));
        })
        return () => unsub();
    }, [])

    return (
        <div className="px-8 py-3 h-screen w-[min(100%, 1400px)] mx-auto flex flex-col justify-center items-center">
            <div className="grid grid-cols-3 lg:grid-cols-4 gap-4 h-full w-full">
                {/* Left Column */}
                <div className="flex flex-col gap-4 ">
                    {/* WEATHER.SYS Panel */}
                    <div className="panel bg-background-secondary">
                        <span className="panel-title">WEATHER</span>
                        <div className="indicator"></div>
                        <div className="mt-8 text-2xl font-bold">24°C</div>
                        <div className="text-sm mt-1">Partly Cloudy</div>
                        <div className="text-xs mt-2">Loc: Delhi, IN</div>
                    </div>

                    {/* SYSTEM.MON & TASKS.EXE Panels */}
                    <div className="flex flex-col gap-4 h-full">
                        <div className="panel flex-1 flex flex-col bg-background-secondary">
                            <span className="panel-title">SYSTEM STATUS</span>
                            <div className="indicator"></div>
                            <div className="flex flex-col gap-2 mt-8 text-xs uppercase">
                                <div className="flex justify-between items-center">
                                    <span>CPU</span>
                                    <span>{cpu}%</span>
                                    <div className="progress-bar w-2/3 h-[4px] rounded-full overflow-hidden bg-background mr-1.5"><div className={`bg-accent h-[5px]`} style={{ width: `${cpu}%` }}></div></div>
                                </div>
                                <div className="flex justify-between items-center">
                                    <span>RAM</span>
                                    <span>{ram}%</span>
                                    <div className="progress-bar w-2/3 h-[4px] rounded-full overflow-hidden bg-background mr-1.5"><div className={`bg-accent h-[5px]`} style={{ width: `${ram}%` }}></div></div>
                                </div>
                                <div className="flex justify-between items-center">
                                    <span>disk</span>
                                    <span>{disk}%</span>
                                    <div className="progress-bar w-2/3 h-[4px] rounded-full overflow-hidden bg-background mr-1.5"><div className={`bg-accent h-[5px]`} style={{ width: `${disk}%` }}></div></div>
                                </div>
                            </div>
                        </div>

                        <div className="panel flex-1 flex flex-col bg-background-secondary">
                            <span className="panel-title">TASKS</span>
                            <div className="indicator"></div>
                            <div className="text-xs mt-8 uppercase">
                                <div className="flex items-center mb-1"><span className="mr-2 text-xl">▶</span>Team meeting @ 14:00</div>
                                <div className="flex items-center mb-1"><span className="mr-2 text-xl">▶</span>Deploy updates @ 16:00</div>
                                <div className="flex items-center mb-1"><span className="mr-2 text-xl">▶</span>Code review pending</div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Middle Column (Main Terminal) */}
                <div className="flex flex-col h-full bg-background-secondary col-span-2 min-h-0">
                    <div className="panel flex-1 flex flex-col h-full min-h-0">
                        <span className="panel-title">TESS V2.1</span>
                        <div className="indicator"></div>

                        {/* Chat Area - Fixed height container */}
                        <div
                            id="chat-messages"
                            className="flex-grow mt-8 mb-4 overflow-y-auto pr-2 min-h-0"
                            style={{ height: '1px' }} // Forces flex-grow to work properly
                        >
                            {chatMessages.map((msg, index) => (
                                <div
                                    key={index}
                                    className={`message-box ${msg.sender === 'user' ? 'user-message-box' : 'ai-message-box'}`}
                                >
                                    {msg.text}
                                </div>
                            ))}
                            <div ref={messagesEndRef} /> {/* auto-scroll anchor */}
                        </div>

                        {/* Input Field */}
                        <form onSubmit={handleChatSubmit} className="flex items-center gap-2 border rounded-lg border-white/10 flex-shrink-0">
                            <input
                                type="text"
                                id="chat-input"
                                placeholder="Enter command..."
                                className="flex-1 p-3 rounded-lg focus:outline-none bg-[var(--color-secondary-bg)] text-[var(--color-foreground)]"
                            />
                            <button type="submit" className="btn mr-2 bg-accent-secondary px-2 py-1 rounded-lg cursor-pointer">Send</button>
                        </form>
                    </div>
                </div>

                {/* Right Column */}
                <div className="flex flex-col gap-4 col-span-3 lg:col-span-1">
                    {/* Navigation */}
                    <Navbar />

                    {/* SYSTEM READY Panel */}
                    <div className="panel">
                        <div className="flex justify-center items-center h-24">
                            <div className="w-16 h-16 relative flex items-center justify-center">
                                {/* Inner pulsing ring */}
                                <div className="absolute inset-0 w-full h-full border rounded-full opacity-75 animate-pulse-ring border-[var(--color-accent)]"></div>
                                {/* Outer circle */}
                                <div className="w-12 h-12 border rounded-full border-[var(--color-accent)]"></div>
                            </div>
                        </div>
                        <div className="text-center mt-2 uppercase">SYSTEM READY</div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default App;
