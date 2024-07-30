use std::io::Write;

use crate::ast::*;
use tract_core::internal::*;
use tract_itertools::Itertools;

macro_rules! comma_loop {
    ($self:ident, $rec: ident, $items: expr) => {
        for (ix, l) in $items.iter().enumerate() {
            if ix > 0 {
                write!($self.w, ", ")?;
            }
            $self.$rec(l)?;
        }
    };
}

pub struct Dumper<'a> {
    nnef: &'a Nnef,
    w: &'a mut dyn std::io::Write,
    with_doc: bool,
}

impl<'a> Dumper<'a> {
    pub fn new(nnef: &'a Nnef, w: &'a mut dyn std::io::Write) -> Dumper<'a> {
        Dumper { nnef, w, with_doc: false }
    }

    pub fn with_doc(mut self) -> Self {
        self.with_doc = true;
        self
    }

    pub fn document(&mut self, document: &Document) -> TractResult<()> {
        writeln!(self.w, "version {};\n", document.version)?;
        for ext in document.extension.iter().sorted() {
            write!(self.w, "extension ")?;
            self.identifier(&ext.0)?;
            writeln!(self.w, " {};", ext.1)?;
        }
        if document.extension.len() > 0 {
            writeln!(self.w)?;
        }
        self.fragments(&document.fragments)?;
        self.graph_def(&document.graph_def)?;
        Ok(())
    }

    pub fn fragments(&mut self, defs: &[FragmentDef]) -> TractResult<()> {
        for fragment_def in defs.iter().sorted_by_key(|frag| &frag.decl.id) {
            self.fragment_def(fragment_def)?
        }
        Ok(())
    }

    pub fn fragment_def(&mut self, def: &FragmentDef) -> TractResult<()> {
        self.fragment_decl(&def.decl)?;
        if let Some(body) = &def.body {
            writeln!(self.w, "\n{{")?;
            for assignment in body {
                self.assignment(assignment)?;
            }
            writeln!(self.w, "}}\n")?;
        } else {
            writeln!(self.w, ";")?;
        };
        Ok(())
    }

    pub(crate) fn fragment_decl(&mut self, decl: &FragmentDecl) -> TractResult<()> {
        write!(self.w, "fragment ")?;
        self.identifier(&decl.id)?;
        if let Some(generic_decl) = &decl.generic_decl {
            if let Some(name) = generic_decl {
                write!(self.w, "<?=")?;
                self.type_name(name)?;
                write!(self.w, ">")?;
            } else {
                write!(self.w, "<?>")?;
            }
        }
        self.parameter_list(&decl.parameters)?;
        write!(self.w, " -> (")?;
        for (ix, res) in decl.results.iter().enumerate() {
            if ix > 0 {
                write!(self.w, ", ")?;
            }
            self.identifier(&res.id)?;
            write!(self.w, ": ")?;
            self.type_spec(&res.spec)?;
        }
        write!(self.w, ")")?;
        Ok(())
    }

    fn parameter_list(&mut self, parameters: &[Parameter]) -> TractResult<()> {
        write!(self.w, "(")?;
        let num_parameters = parameters.len();
        for (ix, param) in parameters.iter().enumerate() {
            if self.with_doc {
                if let Some(doc) = &param.doc {
                    write!(self.w, "\n    # {doc}")?;
                }
            }
            write!(self.w, "\n    ")?;
            self.identifier(&param.id)?;
            write!(self.w, ": ")?;
            self.type_spec(&param.spec)?;
            if let Some(lit) = &param.lit {
                write!(self.w, " = ")?;
                self.literal(lit)?;
            }
            if ix < num_parameters - 1 {
                write!(self.w, ",")?;
            }
        }
        write!(self.w, "\n)")?;
        Ok(())
    }

    fn type_name(&mut self, name: &TypeName) -> TractResult<()> {
        let s = match name {
            TypeName::Integer => "integer",
            TypeName::Scalar => "scalar",
            TypeName::Logical => "logical",
            TypeName::String => "string",
            #[cfg(feature = "complex")]
            TypeName::Complex => "complex",
            TypeName::Any => "?",
        };
        write!(self.w, "{s}")?;
        Ok(())
    }

    fn type_spec(&mut self, spec: &TypeSpec) -> TractResult<()> {
        match spec {
            TypeSpec::Array(t) => {
                self.type_spec(t)?;
                write!(self.w, "[]")?;
            }
            TypeSpec::Single(s) => self.type_name(s)?,
            TypeSpec::Tensor(t) => {
                write!(self.w, "tensor<")?;
                self.type_name(t)?;
                write!(self.w, ">")?;
            }
            TypeSpec::Tuple(types) => {
                write!(self.w, "(")?;
                comma_loop!(self, type_spec, types);
                write!(self.w, ")")?;
            }
        }
        Ok(())
    }

    fn literal(&mut self, lit: &Literal) -> TractResult<()> {
        match lit {
            Literal::Array(lits) => {
                write!(self.w, "[")?;
                comma_loop!(self, literal, lits);
                write!(self.w, "]")?;
            }
            Literal::Logical(b) => write!(self.w, "{}", if *b { "true" } else { "false" })?,
            Literal::Numeric(num) => write!(self.w, "{num}")?,
            Literal::String(s) => write!(self.w, "{s:?}")?,
            Literal::Tuple(lits) => {
                write!(self.w, "(")?;
                comma_loop!(self, literal, lits);
                write!(self.w, ")")?;
            }
        }
        Ok(())
    }

    fn graph_def(&mut self, def: &GraphDef) -> TractResult<()> {
        write!(self.w, "graph ")?;
        self.identifier(&def.id)?;
        write!(self.w, "(")?;
        for (ix, id) in def.parameters.iter().enumerate() {
            if ix > 0 {
                write!(self.w, ", ")?;
            }
            self.identifier(id)?;
        }
        write!(self.w, ") -> (")?;
        for (ix, id) in def.results.iter().enumerate() {
            if ix > 0 {
                write!(self.w, ", ")?;
            }
            self.identifier(id)?;
        }
        writeln!(self.w, ") {{")?;
        for assignment in &def.body {
            self.assignment(assignment)?;
        }
        writeln!(self.w, "}}")?;
        Ok(())
    }

    fn assignment(&mut self, assignment: &Assignment) -> TractResult<()> {
        write!(self.w, "  ")?;
        self.lvalue(&assignment.left)?;
        write!(self.w, " = ")?;
        self.rvalue(&assignment.right)?;
        writeln!(self.w, ";")?;
        Ok(())
    }

    fn lvalue(&mut self, left: &LValue) -> TractResult<()> {
        match left {
            LValue::Identifier(s) => self.identifier(s)?,
            LValue::Tuple(s) => {
                write!(self.w, "( ")?;
                comma_loop!(self, lvalue, s);
                write!(self.w, " )")?;
            }
            LValue::Array(s) => {
                write!(self.w, "[ ")?;
                comma_loop!(self, lvalue, s);
                write!(self.w, " ]")?;
            }
        }
        Ok(())
    }

    pub fn rvalue(&mut self, rv: &RValue) -> TractResult<()> {
        match rv {
            RValue::Array(vals) => {
                write!(self.w, "[")?;
                comma_loop!(self, rvalue, vals);
                write!(self.w, "]")?;
            }
            RValue::Binary(left, op, right) => {
                write!(self.w, "(")?;
                self.rvalue(left)?;
                write!(self.w, " {op} ")?;
                self.rvalue(right)?;
                write!(self.w, ")")?;
            }
            RValue::Comprehension(comp) => self.comprehension(comp)?,
            RValue::Identifier(id) => self.identifier(id)?,
            RValue::IfThenElse(ifte) => {
                self.rvalue(&ifte.then)?;
                write!(self.w, " if ")?;
                self.rvalue(&ifte.cond)?;
                write!(self.w, " else ")?;
                self.rvalue(&ifte.otherwise)?;
            }
            RValue::Invocation(inv) => self.invocation(inv)?,
            RValue::Literal(lit) => self.literal(lit)?,
            RValue::Subscript(left, s) => {
                self.rvalue(left)?;
                write!(self.w, "[")?;
                match s.as_ref() {
                    Subscript::Single(s) => self.rvalue(s)?,
                    Subscript::Range(a, b) => {
                        if let Some(it) = a {
                            self.rvalue(it)?;
                        }
                        write!(self.w, ":")?;
                        if let Some(it) = b {
                            self.rvalue(it)?;
                        }
                    }
                }
                write!(self.w, "]")?;
            }
            RValue::Tuple(vals) => {
                write!(self.w, "(")?;
                comma_loop!(self, rvalue, vals);
                write!(self.w, ")")?;
            }
            RValue::Unary(op, rv) => {
                write!(self.w, "{op}")?;
                self.rvalue(rv)?;
            }
        }
        Ok(())
    }

    fn invocation(&mut self, inv: &Invocation) -> TractResult<()> {
        self.identifier(&inv.id)?;
        if let Some(tn) = &inv.generic_type_name {
            write!(self.w, "<")?;
            self.type_name(tn)?;
            write!(self.w, ">")?;
        }
        write!(self.w, "(")?;
        for (ix, arg) in inv.arguments.iter().enumerate() {
            if ix > 0 {
                write!(self.w, ", ")?;
            }
            if let Some(n) = &arg.id {
                self.identifier(n)?;
                write!(self.w, " = ")?;
            }
            self.rvalue(&arg.rvalue)?;
        }
        write!(self.w, ")")?;
        Ok(())
    }

    fn comprehension(&mut self, comp: &Comprehension) -> TractResult<()> {
        write!(self.w, "[ for")?;
        for iter in &comp.loop_iters {
            self.identifier(&iter.0)?;
            write!(self.w, " in ")?;
            self.rvalue(&iter.1)?;
        }
        if let Some(filter) = &comp.filter {
            write!(self.w, " if ")?;
            self.rvalue(filter)?;
        }
        write!(self.w, " yield ")?;
        self.rvalue(&comp.yields)?;
        write!(self.w, "]")?;
        Ok(())
    }

    fn identifier(&mut self, id: &Identifier) -> TractResult<()> {
        write_identifier(&mut self.w, id, self.nnef.allow_extended_identifier_syntax, false)
    }
}

pub fn write_identifier(
    w: &mut dyn Write,
    id: &Identifier,
    allow_extended_identifier_syntax: bool,
    force_double_quotes: bool,
) -> TractResult<()> {
    if id.0.len() == 0 {
        return Ok(());
    }
    let first = id.0.chars().next().unwrap();
    let force_double_quotes = if force_double_quotes { "\"" } else { "" };
    if (first.is_alphabetic() || first == '_')
        && id.0.chars().all(|c| c.is_alphanumeric() || c == '_')
    {
        write!(w, "{force_double_quotes}{}{force_double_quotes}", id.0)?;
    } else if allow_extended_identifier_syntax {
        write!(w, "i\"{}\"", id.0.replace('\\', "\\\\").replace('\"', "\\\""))?;
    } else {
        write!(w, "{force_double_quotes}")?;
        if !(first.is_alphabetic() || first == '_') {
            write!(w, "_")?;
        }
        for c in id.0.chars() {
            if c.is_alphanumeric() {
                write!(w, "{c}")?;
            } else {
                write!(w, "_")?;
            }
        }
        write!(w, "{force_double_quotes}")?;
    }
    Ok(())
}
