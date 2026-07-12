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
  
  return NextResponse.json({ status: 'cleared' })
}
