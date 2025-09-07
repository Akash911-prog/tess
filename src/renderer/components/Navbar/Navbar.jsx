import { NavLink } from 'react-router-dom'
import ThemeToggleButton from '../ThemeToggleButton/ThemeToggleButton';

const Navbar = () => {

    return (
        <nav className='flex gap-2 bg-background-secondary w-fit px-2 py-1'>
            <NavLink to={'/'} end className={({ isActive }) => isActive ? 'bg-glass/10 px-1 text-accent' : 'px-1'}>
                Home
            </NavLink>
            <NavLink to={'/settings'} end className={({ isActive }) => isActive ? 'bg-glass/10 px-1 text-accent' : 'px-1'}>
                Settings
            </NavLink>
            <ThemeToggleButton />
        </nav>
    )
}

export default Navbar
