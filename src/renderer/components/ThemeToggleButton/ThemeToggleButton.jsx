import { useTheme } from '../../contexts/ThemeProvider'


export default function ThemeToggleButton() {
    const { darkMode, setDarkMode } = useTheme();

    return (
        <button
            onClick={() => setDarkMode(!darkMode)}
            className="relative w-16 h-6 flex items-center rounded-md p-[2px] border-2 border-accent cursor-pointer"
        >
            {/* Background inside */}
            <div className="absolute inset-[2px] bg-background rounded-sm" />

            {/* Thumb */}
            <div
                className={`absolute  w-6 h-4 rounded-sm transition-transform duration-300 ${darkMode ? "translate-x-8 bg-accent" : "translate-x-0 bg-accent"
                    }`}
            />
        </button>
    );
}
