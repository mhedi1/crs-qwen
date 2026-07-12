/** @type {import('next').NextConfig} */
const nextConfig = {
  typescript: { ignoreBuildErrors: true },
  images: { unoptimized: true },
  devIndicators: {
    appIsrStatus: false,
    buildActivity: false,
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://sinbad2ia.ujaen.es:5000/api/:path*',
      }
    ]
  }
}
export default nextConfig
