/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        ink:    '#050B14',
        ink1:   '#07111D',
        panel:  '#0A1523',
        panel2: '#0E1C2C',
        panel3: '#132337',
        line:   '#1A2C40',
        line2:  '#142437',
        teal:   '#20DCE8',
        'teal-d': '#10383F',
        vio:    '#A78BFA',
        'vio-d': '#2B2452',
        coral:  '#FF7B6B',
        amber:  '#F6B84A',
        'amber-d': '#3D2910',
        t1:     '#E8F0FA',
        t2:     '#A9B9CD',
        t3:     '#8197B2',
        t4:     '#526B86',
        ok:     '#41D98C',
        err:    '#F06565',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      fontSize: {
        '2xs': ['0.6875rem', { lineHeight: '1rem' }],
        xs:    ['0.75rem',   { lineHeight: '1.125rem' }],
        sm:    ['0.8125rem', { lineHeight: '1.25rem' }],
      },
      animation: {
        'in':     'fadeIn 0.18s ease',
        'up':     'slideUp 0.18s ease',
        'blink':  'blink 1.8s ease-in-out infinite',
      },
      keyframes: {
        fadeIn:  { from: { opacity: '0' },                              to: { opacity: '1' } },
        slideUp: { from: { opacity: '0', transform: 'translateY(6px)' }, to: { opacity: '1', transform: 'translateY(0)' } },
        blink:   { '0%,100%': { opacity: '1' }, '50%': { opacity: '0.25' } },
      },
    },
  },
  plugins: [],
}
