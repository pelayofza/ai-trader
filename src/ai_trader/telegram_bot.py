import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

load_dotenv()


async def ping(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Hi! I'm online")


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")

    if not token:
        raise RuntimeError(
            "Falta TELEGRAM_BOT_TOKEN en el archivo .env"
        )

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("ping", ping))

    print("Argos iniciado correctamente...")
    app.run_polling()


if __name__ == "__main__":
    main()