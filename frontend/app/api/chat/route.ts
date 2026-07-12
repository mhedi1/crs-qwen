import { NextRequest, NextResponse } from 'next/server'

export async function POST(request: NextRequest) {
  const body = await request.json()
  
  const response = await fetch(
    'http://127.0.0.1:5000/api/chat',
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Cookie': request.headers.get('cookie') || ''
      },
      body: JSON.stringify(body)
    }
  )
  
  const data = await response.json()
  
  const res = NextResponse.json(data)
  
  const setCookie = response.headers.get('set-cookie')
  if (setCookie) {
    res.headers.set('set-cookie', setCookie)
  }
  
  return res
}
