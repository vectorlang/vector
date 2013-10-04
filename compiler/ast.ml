type operator = Add | Sub | Mul | Div ;;

type expr =
    Binop of expr * operator * expr
  | Seq of expr * expr
  | Assign of expr * expr
  | IntLit of int
  | Ident of string ;;
