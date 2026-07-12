/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  trailingSlash: true,
  typescript: { ignoreBuildErrors: true },
  images: { unoptimized: true },
  devIndicators: {
    appIsrStatus: false,
    buildActivity: false,
  }
}
export default nextConfig
