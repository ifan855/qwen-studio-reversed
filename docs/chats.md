# Chats: create, list, read, delete

Everything lives on `q.chats`, a `ChatService` bound to your client.

```python
from qwen_studio import QwenStudio

q = QwenStudio.from_credentials("you@example.com", "password")
MODEL = q.list_model_ids()[0]   # ids rotate - always pick from the catalogue
```

## Create

```python
chat = q.chats.create(MODEL)
print(chat.id)          # server-assigned uuid
```

`POST /chats/new` mirrors the web client body: millisecond timestamp, empty
`chatId` (the server assigns the real id), the target model, `chat_type`
(default `t2t` for text-to-text) and `chat_mode` (default `normal`).

A chat record is empty until the first completion — creating it does *not*
invoke any model.

## List and read

```python
for c in q.chats.list(page=1):          # GET /chats/?page=1&exclude_project=true
    print(c.id, c.title)

chat = q.chats.get(chat.id)             # full record incl. message tree
for m in q.chats.messages(chat.id):     # convenience wrapper
    print(m.role, ":", (m.content or "")[:80])
```

The message list is a tree (nodes carry `parent_id` / `childrenIds`); the
convenience view returns it flat in stored order, with the phase under
which each message was produced.

## Delete — single and batch

```python
q.chats.delete(chat.id)                 # DELETE /chats/{id}
q.chats.delete_many([id1, id2, id3])    # POST /chats/batch_delete {"ids": [...]}
```

Both verified live: `DELETE /chats/{id}` answers
`{"success": true, "data": {"status": true}}` and a subsequent
`GET /chats/{id}` returns the application-level `Not_Found` code (surfaced
by the library as `qwen_studio.exceptions.NotFoundError`).

## Extras

`rename`, `pin`, `archive` and `pinned()` wrap the corresponding
`/chats/...` endpoints observed in the bundle
(`/chats/{id}/title`, `/chats/{id}/pin`, `/chats/{id}/archive`,
`/chats/pinned`).

## Error mapping

| Server reply | Library exception |
|---|---|
| `data.code: "Not_Found"` | `NotFoundError` |
| `data.code: "Bad_Request"` | `BadRequestError` |
| `data.code: "RateLimited"`, `"ParallelLimited"`, `"Too_Many_Requests"` | `RateLimitedError` |
| `data.code: "quotaLimited"`, `"ExceedLimit"`, `"quota_exhausted"` | `QuotaError` |
| `success: false` (anything else) | `APIError` |
| anti-bot punish page (`RGV587`) | `PunishedError` |
