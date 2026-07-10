from agents.Analyzer import Analyzer
import asyncio
analyzer = Analyzer()

async def main():
    await analyzer.stream_run("比较EchoGS和EAP-GS在稀疏视角重建上的方法")

if __name__ == '__main__':
    asyncio.run(main())