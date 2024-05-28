using System;
using System.ComponentModel.DataAnnotations;

namespace Data
{
    public class DataSetModel
    {
        [Key]
        public int Id { get; set; }

        public string Data { get; set; }
    }
}
