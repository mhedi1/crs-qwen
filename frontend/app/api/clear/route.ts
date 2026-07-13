import { NextRequest, NextResponse } from 'next/server'

export async function POST(request: NextRequest) {
  const response = await fetch(
    'http://127.0.0.1:5000/api/clear',
    {
      method: 'POST',
      headers: {
        'Cookie': request.headers.get('cookie') || ''
      }
    }
  )
  
  const res = NextResponse.json({ status: 'cleared' })
  
  const setCookie = response.headers.get('set-cookie')
  if (setCookie) {
    res.headers.set('set-cookie', setCookie)
  }
  
  return res
}
