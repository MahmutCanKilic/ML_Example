using Data;
using Microsoft.EntityFrameworkCore;
using System;

namespace DataAccess
{
    public class DataSetDbContext : DbContext
    {

        public DbSet<DataSetModel> DataSetModels { get; set; }
        protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
        {


            string mySqlUri = Environment.GetEnvironmentVariable("MySqlConnection");
            mySqlUri = "server=localhost;port=13032;database=Account;user=root;password=root;Max Pool Size=200;Min Pool Size=10;Pooling=true";
            optionsBuilder.UseMySql(mySqlUri, ServerVersion.AutoDetect(mySqlUri), p => p.CommandTimeout(600));
            optionsBuilder.UseQueryTrackingBehavior(QueryTrackingBehavior.NoTracking);


            base.OnConfiguring(optionsBuilder);
        }
    }
}
